#!/usr/bin/env python3
"""Check that src/b3_core/SKILL.md is the repo-root SKILL.md.

The packaged path is a symlink in this checkout. ``--check`` fails when the
link points somewhere else, or when a real copy has drifted.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "SKILL.md"
PACKAGED = ROOT / "src" / "b3_core" / "SKILL.md"


def main(argv: list[str]) -> int:
    check = "--check" in argv
    if not SOURCE.is_file():
        print(f"missing {SOURCE}", file=sys.stderr)
        return 1
    if PACKAGED.is_symlink():
        if PACKAGED.resolve() != SOURCE.resolve():
            print(
                f"{PACKAGED} points at {PACKAGED.resolve()}, expected {SOURCE}",
                file=sys.stderr,
            )
            return 1
        print(f"OK {PACKAGED} -> {SOURCE.name}")
        return 0
    text = SOURCE.read_bytes()
    if check:
        if not PACKAGED.is_file() or PACKAGED.read_bytes() != text:
            print("SKILL.md copies differ; keep src/b3_core/SKILL.md in sync", file=sys.stderr)
            return 1
        print("OK SKILL.md copies match")
        return 0
    PACKAGED.write_bytes(text)
    print(f"copied {SOURCE} -> {PACKAGED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
