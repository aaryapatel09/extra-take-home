"""UTF-8 input and atomic local output shared by batch and interactive CLIs."""

import json
import os
import tempfile
from pathlib import Path

from .schema import InputError


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path):
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line, raw in enumerate(stream, 1):
            if raw.strip():
                try:
                    records.append(json.loads(raw))
                except json.JSONDecodeError as exc:
                    raise InputError(f"invalid JSON on line {line}: {exc.msg}") from exc
    return records


def save(path, content):
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent, delete=False
        ) as f:
            temp = Path(f.name)
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        temp.replace(target)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
