from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_checkpoint(
    checkpoints_dir: str | Path,
    experiment_id: str,
    phase: int,
    payload: dict[str, Any],
) -> Path:
    out_dir = Path(checkpoints_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"phase{phase}_{experiment_id}.json"
    if path.exists():
        raise FileExistsError(f"Checkpoint already exists: {path}")
    body = {
        "phase": phase,
        "experiment_id": experiment_id,
        "status": "PASSED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)
    return path
