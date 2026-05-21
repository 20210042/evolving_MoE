from __future__ import annotations

from typing import Any


class ProtoDecodeError(ValueError):
    pass


def read_varint(data: bytes, pos: int) -> tuple[int, int]:
    shift = 0
    result = 0
    while pos < len(data):
        byte = data[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7
        if shift > 70:
            raise ProtoDecodeError("varint too long")
    raise ProtoDecodeError("unexpected end of varint")


def parse_message(data: bytes) -> dict[int, list[Any]]:
    pos = 0
    fields: dict[int, list[Any]] = {}
    while pos < len(data):
        key, pos = read_varint(data, pos)
        field_number = key >> 3
        wire_type = key & 0x07
        if field_number <= 0:
            raise ProtoDecodeError("invalid field number")
        if wire_type == 0:
            value, pos = read_varint(data, pos)
        elif wire_type == 1:
            value = data[pos : pos + 8]
            pos += 8
        elif wire_type == 2:
            length, pos = read_varint(data, pos)
            value = data[pos : pos + length]
            pos += length
        elif wire_type == 5:
            value = data[pos : pos + 4]
            pos += 4
        else:
            raise ProtoDecodeError(f"unsupported wire type {wire_type}")
        fields.setdefault(field_number, []).append(value)
    return fields


def as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def first_text(fields: dict[int, list[Any]], field_number: int) -> str | None:
    values = fields.get(field_number) or []
    if not values:
        return None
    return as_text(values[0])


def first_int(fields: dict[int, list[Any]], field_number: int) -> int | None:
    values = fields.get(field_number) or []
    if not values:
        return None
    value = values[0]
    return value if isinstance(value, int) else None
