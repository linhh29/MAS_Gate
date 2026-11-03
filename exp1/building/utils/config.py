"""Paths and defaults for the building package (repo-root relative)."""

from __future__ import annotations

import os
from pathlib import Path


def print_traces_enabled() -> bool:
    return os.environ.get("MASKILL_PRINT_TRACES", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


# building/utils/config.py -> parents[2] = MAS_Gate repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
BUILDING_HOME = REPO_ROOT / "building"
INIT_SKILL_DIR = BUILDING_HOME / "init_skill"
OPTIMIZED_SKILL_DIR = BUILDING_HOME / "optimized_skill"


def resolve_init_skill_path(explicit: Path | None) -> Path:
    """
    Resolve a skill file path for MAS build.

    Accepts either a directory containing ``SKILL.md`` or a direct ``*.md`` skill file
    (e.g. ``building/optimized_skill/bcp.md``).
    """
    if explicit is not None:
        p = explicit.resolve()
        if p.is_file():
            return p
        if not p.is_dir():
            raise FileNotFoundError(p)
        skill_md = p / "SKILL.md"
        if skill_md.is_file():
            return skill_md
        raise FileNotFoundError(f"Expected SKILL.md or *.md skill file under: {p}")
    default = INIT_SKILL_DIR / "SKILL.md"
    if default.is_file():
        return default.resolve()
    raise FileNotFoundError(
        f"Expected SKILL.md under {INIT_SKILL_DIR}. "
        "Pass --init-skill /path/to/SKILL.md or /path/to/skill.md."
    )
