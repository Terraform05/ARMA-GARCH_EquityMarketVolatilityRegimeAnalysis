from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil
import uuid


@dataclass
class RunRecord:
    run_id: str
    started_at_utc: str
    ended_at_utc: str
    status: str
    data_end_date: str
    objective: str
    evaluation: str
    drawdown_cap: str
    cost_bps: str
    frozen_config_path: str
    reports_dir: str
    notes: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def create_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_run_registry(
    registry_path: Path,
    record: RunRecord,
) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not registry_path.exists()
    with registry_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=asdict(record).keys())
        if new_file:
            writer.writeheader()
        writer.writerow(asdict(record))


def snapshot_config(source: Path, dest_dir: Path, run_id: str) -> Path | None:
    if not source.exists():
        return None
    ensure_dir(dest_dir)
    dest = dest_dir / f"{source.stem}_{run_id}{source.suffix}"
    shutil.copy2(source, dest)
    return dest

