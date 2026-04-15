"""Subprocess helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path


def run_command(command: list[str], *, cwd: str | Path | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)
