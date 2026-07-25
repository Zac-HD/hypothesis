# Proposal: fully pin our test dependencies (#4494)

Status: proposal, for discussion. Delete this file before merging any implementation.

## 1. The problem, precisely

We claim to pin our dependencies, and the lockfiles in `requirements/` do pin the
things they cover. But a large fraction of what CI actually installs never goes
through a lockfile at all, so an upstream release can break `master` at any time
with no commit of ours in between — which is what happened in #4493
(`click==8.2.2` broke the ghostwriter).

The leaks are all in the ad-hoc `pip install` calls that the test scripts and
`tox.ini` make *after* the locked environment has been created:

| Site | Leak |
| --- | --- |
| `hypothesis/scripts/basic-test.sh` | 20 `pip install` lines between the two scripts; ~8 of them install unpinned (`pip install ".[lark]"`, `pip install attrs`, `pip install scipy`, `pip install pytest-repeat`, `pip install .[django]`, `pip install ".[codemods,cli]"`, `pip install ".[pytz, dateutil, zoneinfo]"`, `pip install ".[dpcontracts]"`) |
| `hypothesis/scripts/other-tests.sh` | same pattern |
| `hypothesis/tox.ini` | `pip install .[zoneinfo]`, `pip install coverage_enable_subprocess`, `pip install -r coverage-mod.txt` |

The other ~12 `pip install` lines *are* pinned, but only by grepping the
lockfile at runtime:

```sh
pip install "$(grep '^redis==' ../requirements/coverage.txt)"
pip install "$(grep -E 'black(==| @)' ../requirements/coverage.txt)"
```

That works, but it is 17 grep invocations across the two scripts plus `tox.ini`,
it silently installs nothing useful if the grep ever misses, and it has grown
hand-maintained special cases that duplicate what a resolver already knows:

```sh
if [ "$(python -c 'import sys; print(sys.version_info[:2] == (3, 10))')" = "True" ] ; then
  # Per NEP-29, this is the last version to support Python 3.10
  pip install numpy==2.2.6
elif [ "$(python -c 'import sys; print(sys.version_info[:2] == (3, 11))')" = "True" ] ; then
  # Per SPEC-0, this is the last version to support Python 3.11
  pip install numpy==2.4.6
else
  pip install "$(grep 'numpy==' ../requirements/coverage.txt)"
fi
```

So there are really two goals, and they can be met by the same mechanism:

- **G1 (the bug):** no CI job may install a version we have not committed.
- **G2 (the cleanup):** stop reimplementing dependency resolution in bash.

## 2. The one constraint that shapes every option

Some of our unpinned installs are unpinned **on purpose** — they exist precisely
to test a version that isn't the locked one:

- `[testenv:py310-oldestnumpy]` installs the *oldest* numpy we claim to support.
- `pip install "$(grep -m 1 -oE 'lark>=([0-9.]+)' pyproject.toml | tr '>' =)"` — same idea for lark.
- `[testenv:pytest54|pytest62|pytest74|pytest84|pytest9]` install old/other pytest.
- `[testenv:py31x-pandasNN]` install specific old pandas/numpy.
- `[testenv:numpy-nightly]` installs a pre-release from the scientific-python nightly index.

Any global "constrain everything" switch will break all of these unless it has an
escape hatch. This is the main thing that distinguishes the options below, and
it's worth deciding deliberately rather than discovering it in CI.

## 3. The shared building block: a union constraints lockfile

Every option except Option D is built on the same new artifact the issue calls
for: `requirements/constraints.txt`, a union-of-everything lockfile, generated
alongside the existing ones by `./build.sh upgrade-requirements`.

I checked that this actually resolves, and it does — and it resolves *better*
than I expected. Using `uv pip compile --universal` over
`coverage.in + crosshair.in + fuzzing.in` plus `hypothesis/pyproject.toml`
(~5s, 77 packages), the numpy line comes out as:

```
numpy==2.2.6 ; python_full_version < '3.11'
numpy==2.4.6 ; python_full_version == '3.11.*'
numpy==2.5.1 ; python_full_version >= '3.12'
```

That is *exactly* the NEP-29 / SPEC-0 ladder currently hand-written in both test
scripts. A universal (marker-carrying) resolution therefore lets us delete those
branches rather than port them. Same for pandas (`2.3.3` on <3.11, `3.0.5` above).

Three things I verified rather than assumed:

1. **pip accepts a marker-split constraints file.** A constraints file with three
   `numpy==X ; python_full_version ...` lines is not a "double requirement"
   error; on 3.11 `pip install -c … numpy` correctly installs 2.4.6. `uv pip` agrees.
2. **`PIP_CONSTRAINT` works as an env var**, so the file can be applied to every
   `pip install` in a script without touching the script. (`uv pip` reads `UV_CONSTRAINT`.)
3. **The `hypothesis==` line must be stripped.** `crosshair.txt` already pins
   `hypothesis==6.158.1` via `hypothesis-crosshair`, and hypofuzz pulls one in
   too. A constraints file containing `hypothesis==<anything>` makes
   `pip install .` fail with `ResolutionImpossible` — I confirmed this against a
   throwaway local package. Generation must drop that entry.

One more finding that affects scope: **`tools.in` should stay out of the union.**
It carries `numpy<2.5` (a workaround for mypy stubs using `type` on 3.10/3.11).
Including it caps the union at `numpy==2.4.6` for 3.12+, silently downgrading
what our numpy tests actually exercise. Keeping the tools lockfile separate costs
nothing — the build venv is already isolated and already fully locked.

## 4. Options

### Option A — union constraints file, applied explicitly with `-c` at each site

Generate `requirements/constraints.txt`; edit every `pip install` in the two
scripts and `tox.ini` to pass `-c ../requirements/constraints.txt`; replace the
grep-based installs with plain `pip install -c … redis fakeredis` etc.

- ✅ Surgical: the deliberate-old-version envs (§2) are simply *not* given `-c`, so they keep working with no special handling.
- ✅ Each install site visibly says whether it is pinned — greppable, reviewable.
- ✅ Deletes all 17 greps and both numpy ladders.
- ❌ ~25 edited lines across three files; a *new* `pip install` added later silently reopens the hole, and nothing catches it.

### Option B — union constraints file, applied globally via `PIP_CONSTRAINT`

Same file, but instead of editing each call site, set it once:

```ini
# hypothesis/tox.ini, [testenv]
setenv =
    PIP_CONSTRAINT = {toxinidir}/../requirements/constraints.txt
```

and clear it (`PIP_CONSTRAINT =`) in the handful of envs from §2 that
deliberately install off-lock versions.

- ✅ **Closed by default**: a future `pip install foo` added to a script is pinned automatically. This is the property that actually fixes #4494 rather than fixing today's instances of it.
- ✅ Smallest diff for the pinning half of the work.
- ✅ Opting out is explicit and local to the env that wants old versions, which reads as documentation.
- ⚠️ Needs care with tox: tox scrubs the environment, so `PIP_CONSTRAINT` must be in `setenv`/`passenv` or it just silently doesn't apply. Worth an assertion in `whole_repo_tests` (see §5).
- ⚠️ Action-at-a-distance: a failing install in a script won't obviously point at a constraints file three directories up.
- ❌ Doesn't by itself remove the greps — but nothing stops us doing the Option A cleanup on top, and I'd recommend exactly that.

### Option C — restructure the scripts so the dance mostly disappears

The `pip install X; pytest tests/X; pip uninstall -y X` sequence exists to prove
that each extra works in isolation. But it's a serial, ~20-step install/uninstall
chain inside a single env, which is why it's both slow and hard to pin.
Alternative: one tox env (and one lockfile section) per extra, each installed
from the lockfile up front, run in parallel by CI.

- ✅ Genuinely fixes the root cause: nothing is installed at test time, so there is nothing to leave unpinned.
- ✅ Much better isolation than `pip uninstall -y` (which does not undo transitive installs — today, e.g., installing and removing `fakeredis` can leave its deps behind and change what later tests see).
- ✅ Parallelisable; the `niche`/`full` jobs are currently long serial tails.
- ❌ Substantially larger change, touches CI matrix definitions, and multiplies env-creation cost unless we lean on uv's cache.
- ❌ Doesn't help `tox.ini`'s own ad-hoc installs without also doing A or B.

I'd treat this as desirable but separable — worth doing *after* the leak is
closed, not as the fix for it.

### Option D — go all-in on uv: `uv.lock` + PEP 735 dependency groups

Replace `requirements/*.in` + pip-tools with dependency groups in
`hypothesis/pyproject.toml` and a single `uv.lock`, and drive envs with
`uv sync --frozen --group coverage`.

- ✅ One universal lockfile by construction; the union file stops being a thing we maintain.
- ✅ `--frozen` makes "CI installed something not in the lock" a hard error, not a possibility.
- ✅ Fastest, and we already require uv in `build.sh` and `tox-uv` in `tox.ini`, so it's not a new dependency.
- ❌ Biggest blast radius: rewrites `compile_requirements`, all `deps =` in `tox.ini`, the fuzzing/release workflows, and contributor instructions in `CONTRIBUTING.rst`.
- ❌ The escape hatches from §2 (old pandas, nightly numpy, old pytest) are less natural under `uv sync` than under `pip install`, so those envs need conflicting-group declarations or to stay on the old path anyway.
- ❌ pip-tools is boring and works. The value here is speed and ergonomics, not correctness — and correctness is what #4494 is about.

Worth doing eventually; not worth coupling to this bugfix.

## 5. Recommendation

**Option B as the mechanism, with Option A's cleanup applied on top, and
`uv pip compile` as the generator.** Concretely:

1. Add `requirements/constraints.in` = `-r coverage.in`, `-r crosshair.in`,
   `-r fuzzing.in` (deliberately *not* `tools.in`, per §3).
2. In `compile_requirements`, compile it **first**, universally, then pass it as
   `-c requirements/constraints.txt` when compiling each of the other lockfiles.
   That makes the per-env lockfiles mutually consistent as a side effect, which
   is currently only true by luck. Strip the `hypothesis==` line from the output.
3. Switch the generator from `pip-compile` to `uv pip compile --universal`.
   Universal resolution is what produces the marker ladder in §3; without it we'd
   have to keep the hand-written version branches. (This does mean re-locking
   everything in one noisy commit — I'd land it as its own PR.)
4. Set `PIP_CONSTRAINT` in `tox.ini`'s `[testenv]` `setenv`, and clear it in
   `py310-oldestnumpy`, `numpy-nightly`, `pytest{54,62,74,84,9}`,
   `py3xx-pandasNN`, and around the deliberate `lark>=` minimum install.
   Also set it in `.github/workflows/fuzz.yml`, which today does a bare
   `pip install hypothesis/[all]`.
5. Delete the 17 lockfile greps and both numpy version ladders from
   `basic-test.sh` / `other-tests.sh`, replacing them with plain
   `pip install redis fakeredis` etc.
6. Add a `whole_repo_tests` check that every `pip install` in `hypothesis/scripts/`
   and `tox.ini` is either covered by an inherited `PIP_CONSTRAINT` or sits in an
   explicitly allowlisted "deliberately unpinned" set. Without this, step 4's
   escape hatches decay into step 1's problem again.

Steps 1–2 alone close #4494. Steps 3, 5 are the cleanup the issue also asks for.
Step 6 is what keeps it closed.

## 6. Risks and open questions

- **Re-lock churn.** Moving from pip-compile to uv will change many pins at once. Landing it separately from the pinning change keeps the diff reviewable and makes bisection possible if something regresses.
- **Constraints don't apply to build requirements.** `pip install .` still resolves `maturin>=1.9.4,<2.0` from PyPI at build time (`PIP_CONSTRAINT` is honoured by pip's build isolation, but only if we also confirm it for the maturin backend). Worth checking explicitly — a bad maturin release is exactly the failure mode we're trying to eliminate.
- **`--only-binary` interactions.** Several existing installs pass `--only-binary=:all:`; constraints compose fine with that, but the universal resolution may select a version with no wheel for an exotic target (32-bit Windows, PyPy, GraalVM). The cross-platform matrix should be run before merging.
- **Do we want the union to cover `tools.in` too?** My recommendation is no, on the numpy evidence in §3 — but the alternative is to move the `numpy<2.5` mypy workaround into a marker (`numpy<2.5 ; python_full_version < '3.12'`) and then unify everything. That's arguably cleaner; it just needs someone to confirm the mypy failure really is 3.10/3.11-only.
- **Should `constraints.txt` be checked for staleness in CI?** `check-not-changed` after a re-lock would catch a hand-edited lockfile, at the cost of a network-dependent job.
