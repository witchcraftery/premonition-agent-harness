from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrialConfig:
    name: str
    dataset: str
    split: str
    input_path: Path
    top_k: int
    notes: str = ""

    @classmethod
    def from_dict(cls, row: dict[str, Any], base_path: Path) -> "TrialConfig":
        input_path = Path(str(row["input_path"]))
        if not input_path.is_absolute():
            input_path = base_path / input_path

        return cls(
            name=str(row["name"]),
            dataset=str(row["dataset"]),
            split=str(row.get("split", "dev")),
            input_path=input_path,
            top_k=int(row.get("top_k", 3)),
            notes=str(row.get("notes", "")),
        )


def load_trial_config(path: Path) -> TrialConfig:
    with path.open("r", encoding="utf-8") as handle:
        return TrialConfig.from_dict(json.load(handle), base_path=path.parent)
