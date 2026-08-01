# Stabilizing observability: final design and migration plan

Status: draft for review. Context: issue #4387 ("choice sequence and spans in
observability + graduate observability from experimental") and PR #4563
("Stabilize observability", open, reviewed but not merged).

## 1. Where we are

**On master today**, observability is a pile of module-level state in
`hypothesis.internal.observability`:

- A per-thread callback registry (`add_observability_callback`,
  `remove_observability_callback`, `with_observability_callback`,
  `observability_enabled`, plus the deprecated `TESTCASE_CALLBACKS` shim), with
  an `all_threads=True` variant that also receives a thread id.
- Two import-time globals driven by env vars:
  `OBSERVABILITY_COLLECT_COVERAGE` (from
  `HYPOTHESIS_EXPERIMENTAL_OBSERVABILITY_NOCOVER`) and `OBSERVABILITY_CHOICES`
  (from `HYPOTHESIS_EXPERIMENTAL_OBSERVABILITY_CHOICES`).
- File delivery: `HYPOTHESIS_EXPERIMENTAL_OBSERVABILITY` registers
  `_deliver_to_file` at import time, and evicts week-old files at import time.
- Backend integration: `PrimitiveProvider.add_observability_callback: ClassVar[bool]`
  opts the provider's `on_observation` method into the callback registry, wired
  up by `ConjectureRunner.observe_for_provider` via `with_observability_callback`.
- Everything user-facing lives under `hypothesis.internal.*` and is documented
  in the "internals" reference page with experimental warnings.

**PR #4563** replaces the registry with configuration-as-data: an
`ObservabilityConfig(coverage, choices, callbacks)` dataclass, a new
`settings(observability=...)` argument accepting `True | False | None |
ObservabilityConfig`, a `HYPOTHESIS_OBSERVABILITY` env var feeding the default
profile, and a union operator (`|`) for combining configs. Delivery becomes
"iterate `settings.observability.callbacks`". The registry functions,
`deliver_observation`, and `TESTCASE_CALLBACKS` are deleted;
`observability_enabled()` is deprecated.

Zac's outstanding review points on #4563:

1. `observe_for_provider` temporarily mutating `self.settings._observability`
   to splice in the backend's callback is "the wrong mechanism".
2. `PrimitiveProvider` should expose an `ObservabilityConfig` as an attribute,
   rather than the `add_observability_callback` bool + implicit callback.
3. The PR is big; pull out independently-mergeable pieces (file-delivery
   eviction; `PrimitiveProvider.observability` as an internal interface).
4. Docs: split a how-to guide out, trim the reference; write a blog post.

The design below keeps #4563's public API essentially intact and resolves
(1)+(2) with one structural idea: **the effective config is computed per test
case and attached to `ConjectureData`**, instead of being read from (or
smuggled into) global settings.

## 2. Final design

### 2.1 Public API surface

```python
from hypothesis import settings, ObservabilityConfig

# simple on/off
@settings(observability=True)          # -> ObservabilityConfig()
@settings(observability=False)         # -> None (disabled)

# fine-grained
@settings(observability=ObservabilityConfig(
    coverage=True,          # include the `coverage` field (default True)
    choices=False,          # include metadata.choice_nodes / .choice_spans (default False)
    callbacks=(...),        # where observations go; defaults to (deliver_to_file,)
))
```

- `hypothesis.ObservabilityConfig` — exported at top level, like `Phase`/`Verbosity`.
- `settings.observability` — validated to `ObservabilityConfig | None`
  (`True` normalizes to `ObservabilityConfig()`; `False`/`None` to `None`).
  Inherits like every other setting; the *default profile* value comes from the
  `HYPOTHESIS_OBSERVABILITY` env var (`1/0/true/false`, case-insensitive) and
  is otherwise `None`.
- A new **public module `hypothesis.observability`** as the stable home for the
  supporting types (implementation can stay in `internal/observability.py` and
  be re-exported):
  - `Observation`, `TestCaseObservation`, `InfoObservation`,
    `ObservationMetadata`, `PredicateCounts` — so callback authors can type and
    destructure observations without importing from `hypothesis.internal`.
  - `deliver_to_file` — the default callback, public and named, so users can
    *compose* it: `callbacks=(deliver_to_file, my_callback)` or drop it:
    `callbacks=(my_callback,)`.
  - (optional, see open questions) `observability_enabled()`.

  Rationale: "stable feature whose docs tell you to import from
  `hypothesis.internal.observability`" is self-contradictory, and we already
  regret that pattern with `PrimitiveProvider`. This is cheap now and very
  annoying to retrofit later.

### 2.2 `ObservabilityConfig` semantics

```python
@dataclass(frozen=True)
class ObservabilityConfig:
    coverage: bool = True
    choices: bool = False
    callbacks: tuple[Callable[[Observation], None], ...] = (deliver_to_file,)
```

- **Frozen.** Configs are shared between settings objects, profiles, and (in
  the engine) unioned copies; aliasing bugs from in-place mutation are not
  worth the convenience. `__post_init__` converts any iterable `callbacks` to a
  tuple via `object.__setattr__`.
- **Union**: `a | b` returns a config with `coverage=a.coverage or b.coverage`,
  `choices=a.choices or b.choices`, and the callbacks of `a` followed by those
  of `b` not already present. `config | None == config`, `None` handled via
  `__ror__`; `True` coerces to `ObservabilityConfig()`. This is the one
  operation the engine needs (settings-level ∪ provider-level), and it is also
  handy for users composing profiles.
- **At least one callback required** at the settings level (validated in
  `_validate_observability`, not in `__post_init__` — see §2.4 for why
  provider-level configs are exempt): enabling observability with nowhere to
  deliver is always a bug.
- Callbacks are called synchronously on the thread that produced the
  observation, in order. A callback needing a thread id calls
  `threading.get_ident()` itself — this replaces the `all_threads=True`
  two-argument form.

### 2.3 The effective config lives on `ConjectureData`

This is the core structural change, and what resolves the "wrong mechanism"
review comment.

Every `ConjectureData` gets an attribute set at construction time:

```python
data.observability: ObservabilityConfig | None
```

computed by whoever creates the data object:

- In `ConjectureRunner`: `settings.observability | provider_config`, where
  `provider_config` is the active backend's contribution (see §2.4) — and only
  when the backend actually generates this test case (i.e. not while
  `_switch_to_hypothesis_provider` is set, and not during shrinking with the
  hypothesis provider). The current callback-time guard in
  `observe_for_provider` becomes structural: test cases the backend didn't
  generate simply never carry its callbacks.
- In `core.py` for the paths that build their own data (explicit examples,
  `reproduce_failure`, final-failure replay): `state.settings.observability`.

Everything that currently asks "is observability on, and what should I
collect?" *during a test case* consults `data.observability` instead of global
state:

- `ConjectureData.draw` recording `_observability_args` (also removes the
  per-draw thread-local `settings()` lookup #4563 currently does),
- `assume()` / `event()` predicate counting in `control.py` (via
  `current_build_context().data`),
- stateful printing in `stateful.py`,
- coverage tracing decision in `StateForActualGivenExecution.execute_once`
  (`data.observability is not None and data.observability.coverage`),
- `make_testcase(..., observability=data.observability)` gating `coverage` and
  `choice_nodes`/`choice_spans`.

Delivery: `deliver_observation(observation, config)` iterates
`config.callbacks`. For test-case observations, `config` is
`data.observability`; for info observations (e.g. the statistics message,
which has no associated data), it is `state.settings.observability`.

Decisions made *outside* any test case (e.g. "wrap the test with repr-capture",
"disable the deadline-less fast path") key off `settings.observability` as in
#4563 — settings are thread-local (`DynamicVariable` over `threading.local`)
and the engine already enters `local_settings(self.settings)`, so this is
per-thread correct without a registry.

Net effect versus #4563: `observe_for_provider` and its
`settings._observability` mutation are deleted outright; nothing ever writes to
settings at runtime.

### 2.4 Backend / provider integration

Replace the bool with a config, per Zac's review:

```python
class PrimitiveProvider(abc.ABC):
    #: Observability options this provider needs while it is generating
    #: test cases. If None, on_observation is never called and the provider
    #: adds nothing to observability. If set, Hypothesis unions this with the
    #: user's settings-level config for every test case this provider
    #: generates, and calls on_observation with each resulting test_case
    #: observation.
    observability: ObservabilityConfig | None = None
```

- Typical backend usage:
  `observability = ObservabilityConfig(coverage=False, choices=True, callbacks=())`
  — "I need choice metadata, I don't need Hypothesis's coverage tracing, and I
  have no extra delivery targets."
- `on_observation` **stays** as the delivery mechanism for providers. The
  engine wires it up when building `data.observability`
  (`provider_config | settings_config`, plus the bound `on_observation` as a
  callback appended by the engine, still filtered to `type == "test_case"`).
  Keeping it as a named method rather than "put a bound method in your own
  callbacks tuple" preserves the existing contract (test-case-only, per-test-
  function lifetime) and means providers don't need `__init__`-time self-
  reference gymnastics. Class-level declaration stays possible since the
  common case is a static config.
- Empty `callbacks=()` is legal in provider-level configs (delivery comes from
  the union with the user's config and/or `on_observation`); this is why the
  ≥1-callback check belongs in settings validation, not in the dataclass.
- `add_observability_callback: ClassVar[bool]` is currently documented but
  provisional: keep it working for one release cycle — if a provider sets it
  and does not set `observability`, treat it as
  `observability = ObservabilityConfig(coverage=False, callbacks=())` and warn.
  Then delete.
- Behavior fix we get for free: today, a backend opting into observations
  flips `observability_enabled()` on globally, which silently enables coverage
  tracing of e.g. crosshair-executed tests. Under this design a backend only
  gets what it asked for.

### 2.5 File delivery

- `deliver_to_file` is public (§2.1) and remains the default callback, writing
  `.hypothesis/observed/<date>_{testcases,info}.jsonl`.
- Eviction of >week-old files moves from import time to once-per-process in
  `ConjectureRunner.run()` (as in #4563) — import-time filesystem traffic on
  every `import hypothesis` was always wrong, and this piece is mergeable
  today with zero API implications.
- The known multi-threaded contention on the single file lock is an explicitly
  deferred follow-up (queue + background writer); not a blocker for
  stabilization, but the docs should note that heavy multi-threaded use may
  prefer a custom callback.

### 2.6 Env vars

| Variable | Fate |
|---|---|
| `HYPOTHESIS_OBSERVABILITY` | **New.** `1/true` → default profile gets `ObservabilityConfig()`; `0/false` → `None`; anything else → error at import. Boolean-only for now; comma-separated tokens (`choices`, `nocover`, …) are an easy backwards-compatible extension later, so we deliberately don't design them in yet. |
| `HYPOTHESIS_EXPERIMENTAL_OBSERVABILITY` | Works + `note_deprecation` for ≥6 months, then removed. |
| `HYPOTHESIS_EXPERIMENTAL_OBSERVABILITY_CHOICES` | Works (maps to `ObservabilityConfig(choices=True)`) + deprecation, then removed. |
| `HYPOTHESIS_EXPERIMENTAL_OBSERVABILITY_NOCOVER` | Removed without deprecation (as in #4563; it was experimental, and its main constituency was pre-3.12 `sys.settrace` overhead). `ObservabilityConfig(coverage=False)` is the replacement. |

### 2.7 What is removed / deprecated, and the migration story

| Today | Final design |
|---|---|
| `add_observability_callback(f)` | `settings(observability=ObservabilityConfig(callbacks=(f, ...)))` — or register on a profile for process-wide effect |
| `add_observability_callback(f, all_threads=True)` | Put it in the default/active profile (settings are per-thread, profiles are process-wide); call `threading.get_ident()` inside the callback if the thread matters |
| `remove_observability_callback` / `with_observability_callback` | Scoped settings: apply per-test, or `settings.load_profile` / `local_settings` |
| `observability_enabled()` | `settings().observability is not None` for "is it on here"; inside a test, `current_build_context().data.observability` is authoritative (includes backend contribution) |
| `TESTCASE_CALLBACKS` (already deprecated) | delete on its existing schedule |
| `OBSERVABILITY_COLLECT_COVERAGE` / `OBSERVABILITY_CHOICES` globals | `ObservabilityConfig(coverage=..., choices=...)`; keep module-level `__getattr__` shims emitting deprecation warnings for one cycle, since HypoFuzz and research code read these |
| `hypothesis.internal.observability.X` imports | `hypothesis.observability.X`; old locations keep working as re-exports (no warning needed for the dataclasses; warnings for the removed registry functions) |

Since the whole feature is provisional we *may* hard-break, but everything
above except `NOCOVER` costs us almost nothing to shim for one deprecation
cycle, and the known consumers are worth being gentle with:

- **Tyche** — file-based; unaffected except env var rename. Coordinate so its
  docs recommend `HYPOTHESIS_OBSERVABILITY=1` (per liam's note on #4387).
- **HypoFuzz** — uses the callback registry and the `OBSERVABILITY_CHOICES` /
  `COLLECT_COVERAGE` globals; we control it, migrate it in lockstep, and its
  needs (in-memory delivery, choices on, coverage via its own tracer) are all
  first-class in the new config.
- **hypothesis-crosshair** — `add_observability_callback = True` provider;
  the one-cycle shim in §2.4 covers it until it sets `observability = ...`.

### 2.8 Data format

Unchanged from master/#4563: `choice_nodes` (type, value, constraints,
was_forced) and `choice_spans` (`[label, start, end, discarded]`) under
`metadata`, present iff `choices=True`. The observation schema is stable;
`coverage`'s *format* and the choices metadata keep an "unstable format"
warning in the schema docs (as #4563 already does) so we can move to
branch-coverage or a more compact encoding later without a major event.
`metadata.notes` (added on master after #4563's last rebase) is kept — rebase
item for the PR.

## 3. Incremental migration plan

Each step is a separately-reviewable PR that leaves the tree green and
releasable. Steps 1–2 are pure refactors (no user-visible change, patch
releases); the public API appears only in step 4.

**Step 1 — file-delivery cleanups** *(small, no API surface; Zac already
suggested pulling this out)*
Move stale-file eviction from import time into `ConjectureRunner.run()`
(once-per-process flag). Drop the unused thread-id parameter from
`_deliver_to_file` by registering it with a wrapper. Mergeable immediately.

**Step 2 — internal `ObservabilityConfig` + effective config on
`ConjectureData`** *(behavior-preserving refactor)*
Add the frozen dataclass with `|` union to `internal/observability.py`
(not exported, not documented). Thread `data.observability` through:
engine/core set it from a config *derived from the existing globals and
registry state* (i.e. `ObservabilityConfig(coverage=OBSERVABILITY_COLLECT_COVERAGE,
choices=OBSERVABILITY_CHOICES, callbacks=<registry>)` when
`observability_enabled()`, else `None`); `make_testcase` gains the
`observability=` parameter; `data.py`/`control.py`/`stateful.py`/`core.py`
consult `data.observability`. Delivery still goes through the existing
registry, so external behavior is identical.

**Step 3 — provider interface** *(internal/unstable, per Zac's suggestion)*
Add `PrimitiveProvider.observability: ObservabilityConfig | None = None`;
engine computes the per-data union and calls `on_observation` through it;
delete `observe_for_provider`'s callback juggling. Shim
`add_observability_callback = True` → implied config, with a warning. Update
crosshair tests; PR to hypothesis-crosshair.

**Step 4 — the public API** *(minor release; the heart of #4563, now much
smaller)*
`settings(observability=...)` + validation, `HYPOTHESIS_OBSERVABILITY` env
var, `hypothesis.ObservabilityConfig` export, public
`hypothesis.observability` module (`deliver_to_file`, observation types), and
switch the default profile / env-var plumbing over. The engine's
"effective config" source becomes `settings.observability` instead of the
registry. The registry functions become thin deprecated wrappers that mutate
the *default profile's* config (preserving observable behavior for one cycle),
`observability_enabled()` gets its deprecation, and the experimental env vars
get theirs. RELEASE.rst announces stabilization.

**Step 5 — docs + comms**
Restructure: a how-to guide (enable it, view with Tyche/pandas, write a
callback, backend integration) and a trimmed reference (settings +
`ObservabilityConfig` + schemas). Blog post announcing stabilization and
research directions. Coordinate Tyche + HypoFuzz doc updates.

**Step 6 — cleanup (≥6 months later)**
Delete the registry wrappers, `TESTCASE_CALLBACKS`, the experimental env
vars, the globals' `__getattr__` shims, and the
`add_observability_callback` provider shim.

The existing #4563 branch gets carved up rather than discarded: step 1 and
most of step 4's settings/validation/docs/tests lift straight out of it;
steps 2–3 are the new work that answers the review feedback.

## 4. Open questions (with recommendations)

1. **Should callbacks live in settings at all?** Settings have so far been
   plain data + the database object; callbacks make profiles less
   printable/comparable. Alternatives: a process-global registry (status quo —
   but then config and delivery live in different places, and per-test scoping
   is clumsy) or a separate `hypothesis.observability.register()` API.
   *Recommendation: keep callbacks in the config.* The database precedent
   covers "behavioral object in settings", per-test/per-profile scoping falls
   out for free, and the engine union gives backends a principled hook. This
   was also #4563's conclusion; Zac's discomfort was specifically with
   *mutating* settings, which §2.3 eliminates.
2. **Frozen vs mutable config** — recommend frozen (§2.2); #4563 currently has
   `frozen=False`.
3. **Name**: `ObservabilityConfig` vs `Observability` vs
   `ObservabilitySettings`. Recommend `ObservabilityConfig` (matches the PR;
   "Settings" invites confusion with `hypothesis.settings`).
4. **Keep `observability_enabled()` public?** It's the ergonomic check for
   strategy/backend authors deciding whether to compute expensive reprs, and
   under this design the *correct* check (data-level, backend-aware) is
   otherwise a mouthful. Recommend: re-home it in `hypothesis.observability`,
   implemented as "current build context's `data.observability`, falling back
   to `settings().observability`" — rather than deprecating it as #4563 does.
5. **Thread-id delivery**: is dropping the `all_threads=True` two-argument
   callback form acceptable for HypoFuzz? (Callbacks run on the producing
   thread, so `threading.get_ident()` recovers it; recommend yes.)
6. **`HYPOTHESIS_OBSERVABILITY` richer values now or later?** Recommend later
   (boolean-only is forward-compatible with token lists).
