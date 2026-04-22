"""Invoke RAITAP's CLI on a Hydra config.

RAITAP exposes a CLI (`raitap ...`) driven by Hydra configs rather than a
traditional Python class API. We shell out so pipeline steps stay thin.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def run_assessment(
    config_dir: Path,
    config_name: str,
    output_dir: Path | None = None,
    extra_overrides: list[str] | None = None,
) -> int:
    """Run `raitap --config-dir <dir> --config-name <name> [overrides]`.

    Returns the process exit code.
    """
    cmd = [
        "raitap",
        "--config-dir",
        str(config_dir),
        "--config-name",
        config_name,
    ]
    if output_dir is not None:
        cmd.append(f"hydra.run.dir={output_dir}")
    if extra_overrides:
        cmd.extend(extra_overrides)
    return subprocess.run(cmd, check=True).returncode
