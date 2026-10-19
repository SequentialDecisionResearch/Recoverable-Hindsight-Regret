from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import AppConfig, dump_config_dict
from .environment import environment_snapshot
from .hashing import sha256_file, sha256_json


@dataclass(frozen=True)
class RunContext:
    experiment_id: str
    run_dir: Path
    manifest_path: Path


def _git_commit(project_root: Path) -> str | None:
    try:
        p = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return p.stdout.strip()
    except Exception:
        return None


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


class ExperimentRegistry:
    def __init__(self, project_root: str | Path, config: AppConfig):
        self.project_root = Path(project_root).resolve()
        self.config = config
        self.outputs_dir = (self.project_root / config.paths.outputs_dir).resolve()
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def start_run(self, config_file: str | Path, data_files: list[Path] | None = None) -> RunContext:
        config_dict = dump_config_dict(self.config)
        config_hash = sha256_json(config_dict)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = uuid.uuid4().hex[:8]
        experiment_id = f"{stamp}_p{self.config.phase}_{config_hash[:8]}_{suffix}"
        run_dir = self.outputs_dir / experiment_id
        if run_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing run directory: {run_dir}")
        for name in ["configs", "logs", "replays", "predictions", "tables", "figures", "models"]:
            (run_dir / name).mkdir(parents=True, exist_ok=True)

        config_file = Path(config_file).resolve()
        shutil.copy2(config_file, run_dir / "configs" / config_file.name)

        data_manifest = []
        for path in sorted(data_files or []):
            path = Path(path).resolve()
            data_manifest.append({
                "path": str(path.relative_to(self.project_root)) if path.is_relative_to(self.project_root) else str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })

        manifest = {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "phase": self.config.phase,
            "status": "STARTED",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "project_root": str(self.project_root),
            "git_commit": _git_commit(self.project_root),
            "config_hash": config_hash,
            "config": config_dict,
            "environment": environment_snapshot(),
            "data_files": data_manifest,
        }
        manifest_path = run_dir / "manifest.json"
        _write_json_atomic(manifest_path, manifest)
        return RunContext(experiment_id, run_dir, manifest_path)

    def finalize_run(self, ctx: RunContext, status: str, extra: dict[str, Any] | None = None) -> None:
        manifest = json.loads(ctx.manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = status
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        if extra:
            manifest.update(extra)
        _write_json_atomic(ctx.manifest_path, manifest)
