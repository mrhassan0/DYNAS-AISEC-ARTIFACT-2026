#!/usr/bin/env python3
"""Generate fixed-regime scenarios from the paper's curriculum definitions.

The source YAML files contain the parameter values used by the three-stage
curriculum. This script derives two additional controls without introducing
new defense parameters:

* ids_off: keep the curriculum's first (defenses-disabled) stage active.
* defended: keep the curriculum's final (full-defense) stage active.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DIR = ROOT / "scenarios"
SOURCE_NAMES = (
    "corp_100hosts_dynamic.v2.yaml",
    "corp_100hosts_dynamic_bridge.v2.yaml",
    "corp_100hosts_dynamic_varA.v2.yaml",
    "corp_100hosts_dynamic_varB.v2.yaml",
)


def fixed_stage(stage: dict, name: str) -> dict:
    result = copy.deepcopy(stage)
    result["name"] = name
    result.pop("start_frac", None)
    result.pop("end_frac", None)
    result["start_epoch"] = 0
    result["end_epoch"] = 999999
    return result


def write_variant(source_path: Path, regime: str, stage: dict) -> None:
    with source_path.open("r", encoding="utf-8") as handle:
        scenario = yaml.safe_load(handle)

    scenario["curriculum"] = {
        "enabled": True,
        "stages": [stage],
    }

    output_dir = SCENARIO_DIR / "regimes" / regime
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / source_path.name
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(scenario, handle, sort_keys=False)


def main() -> None:
    for name in SOURCE_NAMES:
        source_path = SCENARIO_DIR / name
        with source_path.open("r", encoding="utf-8") as handle:
            scenario = yaml.safe_load(handle)

        stages = scenario["curriculum"]["stages"]
        if len(stages) < 2:
            raise ValueError(f"{source_path} does not contain a staged curriculum")

        write_variant(
            source_path,
            "ids_off",
            fixed_stage(stages[0], "ids_off"),
        )
        write_variant(
            source_path,
            "defended",
            fixed_stage(stages[-1], "full_defense_from_start"),
        )

    print(f"Generated fixed-regime scenarios under {SCENARIO_DIR / 'regimes'}")


if __name__ == "__main__":
    main()
