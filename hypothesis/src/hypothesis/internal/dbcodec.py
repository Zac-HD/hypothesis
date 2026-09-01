# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Encodings shared by the database backends.

See ``guides/database-design.md`` for the reasoning behind these formats.
"""

import struct
import time
from hashlib import blake2b
from typing import Literal, NamedTuple, TypeAlias

from hypothesis.errors import InvalidArgument

KeyPartT: TypeAlias = bytes | str | int
KeyTupleT: TypeAlias = tuple[KeyPartT, ...]

_BYTES, _STR, _INT = 1, 2, 3
_EXACT_TYPES = frozenset({bytes, str, int})
_INT_BIAS = 1 << 63

# Legacy keys starting with this prefix hold structured data, when structured data
# is stored through the old save / fetch / delete methods.
MAGIC = b"\x00hypothesis\x01"
_FIELD, _INDEX, _LOG = b"m", b"i", b"l"


def _pack_uleb128(value: int) -> bytes:
    """
    Serialize an integer into variable-length bytes. For each byte, the first 7
    bits represent (part of) the integer, while the last bit indicates whether the
    integer continues into the next byte.

    https://en.wikipedia.org/wiki/LEB128
    """
    parts = bytearray()
    assert value >= 0
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            byte |= 0x80
        parts.append(byte)
        if not value:
            return bytes(parts)


def _unpack_uleb128(buffer: bytes, start: int = 0) -> tuple[int, int]:
    """Inverts _pack_uleb128, returning (bytes consumed, value)."""
    value = 0
    for i, byte in enumerate(buffer[start:]):
        value |= (byte & 0x7F) << (i * 7)
        if not byte >> 7:
            return (i + 1, value)
    raise ValueError("truncated uleb128")


def as_tuple(value: object, *, what: str, allow_empty: bool) -> KeyTupleT:
    """Normalise a key or field to a tuple of bytes, str, and int components."""
    parts = value if isinstance(value, tuple) else (value,)
    if not parts and not allow_empty:
        raise InvalidArgument(f"{what} must not be empty")
    if all(type(part) in _EXACT_TYPES for part in parts):
        return parts
    out: list[KeyPartT] = []
    for part in parts:
        if isinstance(part, bool):
            part = int(part)
        elif isinstance(part, (bytearray, memoryview)):
            part = bytes(part)
        elif not isinstance(part, (bytes, str, int)):
            raise InvalidArgument(
                f"{what} components must be bytes, str, or int, not {part!r}"
            )
        out.append(part)
    return tuple(out)


def is_legacy(key: KeyTupleT) -> bool:
    """A key whose only component is bytes holds a set, like the old interface."""
    return len(key) == 1 and isinstance(key[0], bytes)


def encode(parts: KeyTupleT) -> bytes:
    out = bytearray()
    for part in parts:
        if isinstance(part, bytes):
            out.append(_BYTES)
            out += _pack_uleb128(len(part))
            out += part
        elif isinstance(part, str):
            data = part.encode("utf-8", "surrogatepass")
            out.append(_STR)
            out += _pack_uleb128(len(data))
            out += data
        else:
            if not -_INT_BIAS <= part < _INT_BIAS:
                raise InvalidArgument(
                    f"integer key components must fit in 64 bits, not {part}"
                )
            out.append(_INT)
            out += (part + _INT_BIAS).to_bytes(8, "big")
    return bytes(out)


def decode(data: bytes) -> KeyTupleT:
    parts: list[KeyPartT] = []
    i = 0
    while i < len(data):
        tag = data[i]
        i += 1
        if tag == _INT:
            parts.append(int.from_bytes(data[i : i + 8], "big") - _INT_BIAS)
            i += 8
            continue
        if tag not in (_BYTES, _STR):
            raise ValueError(f"unknown tag {tag} in encoded key")
        used, size = _unpack_uleb128(data, i)
        i += used
        chunk = data[i : i + size]
        if len(chunk) != size:
            raise ValueError("truncated key component")
        i += size
        parts.append(chunk if tag == _BYTES else chunk.decode("utf-8", "surrogatepass"))
    return tuple(parts)


def has_prefix(encoded: bytes, encoded_prefix: bytes) -> bool:
    # Components are self-delimiting, so a byte prefix is a tuple prefix.
    return encoded.startswith(encoded_prefix)


def short_hash(data: bytes) -> bytes:
    return blake2b(data, digest_size=16).digest()


# Reserved legacy keys, used to store structured data through the old methods.


def field_key(ek: bytes, ef: bytes) -> bytes:
    return MAGIC + _FIELD + _pack_uleb128(len(ek)) + ek + ef


def index_key(ek: bytes) -> bytes:
    return MAGIC + _INDEX + ek


def log_key(ek: bytes) -> bytes:
    return MAGIC + _LOG + ek


class LegacyKey(NamedTuple):
    kind: Literal["set", "field", "index", "log"]
    key: KeyTupleT
    field: KeyTupleT | None = None


def parse_legacy_key(raw: bytes) -> LegacyKey:
    """Interpret a key passed to save, fetch, or delete."""
    if raw.startswith(MAGIC) and len(raw) > len(MAGIC):
        kind, body = raw[len(MAGIC) : len(MAGIC) + 1], raw[len(MAGIC) + 1 :]
        try:
            if kind == _FIELD:
                used, size = _unpack_uleb128(body)
                key = decode(body[used : used + size])
                field = decode(body[used + size :])
                if key and not is_legacy(key):
                    return LegacyKey("field", key, field)
            elif kind in (_INDEX, _LOG):
                key = decode(body)
                if key and not is_legacy(key):
                    return LegacyKey("index" if kind == _INDEX else "log", key)
        except ValueError:
            pass
    return LegacyKey("set", (raw,))


# Log entry ids: 8 bytes of milliseconds, then an 8-byte sequence number.

ENTRY_ID_SIZE = 16


def make_entry_id(ms: int, seq: int) -> bytes:
    return struct.pack(">QQ", ms, seq)


def split_entry_id(entry_id: bytes) -> tuple[int, int]:
    if len(entry_id) != ENTRY_ID_SIZE:
        raise InvalidArgument(
            f"log entry ids are {ENTRY_ID_SIZE} bytes, got {entry_id!r}"
        )
    return struct.unpack(">QQ", entry_id)


def next_entry_id(last: bytes | None, now_ms: int) -> bytes:
    """The smallest id after ``last`` that is not before ``now_ms``."""
    last_ms, last_seq = split_entry_id(last) if last else (-1, 0)
    if now_ms > last_ms:
        return make_entry_id(now_ms, 0)
    return make_entry_id(last_ms, last_seq + 1)


# Journal cursors: when the cursor was issued, then a backend-specific position.


def make_cursor(position: bytes, issued_at: float | None = None) -> bytes:
    return struct.pack(">d", time.time() if issued_at is None else issued_at) + position


def split_cursor(cursor: bytes) -> tuple[float, bytes]:
    if not isinstance(cursor, bytes) or len(cursor) < 8:
        raise InvalidArgument(f"invalid journal cursor {cursor!r}")
    return struct.unpack(">d", cursor[:8])[0], cursor[8:]
