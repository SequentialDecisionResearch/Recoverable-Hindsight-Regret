from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(obj: Any) -> bytes:
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return text.encode("utf-8")


def sha256_json(obj: Any) -> str:
    return sha256_bytes(canonical_json_bytes(obj))
