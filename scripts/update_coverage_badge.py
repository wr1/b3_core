#!/usr/bin/env python3
"""Write shields.io endpoint JSON from coverage.json (pytest-cov)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COVERAGE_JSON = ROOT / "coverage.json"
BADGE_PATH = ROOT / "badges" / "coverage.json"


def main() -> int:
    if not COVERAGE_JSON.is_file():
        print(
            f"missing {COVERAGE_JSON}; run: pytest --cov-report=json:coverage.json",
            file=sys.stderr,
        )
        return 1
    pct = float(json.loads(COVERAGE_JSON.read_text())["totals"]["percent_covered"])
    msg = f"{pct:.0f}%"
    if pct >= 80:
        color = "brightgreen"
    elif pct >= 70:
        color = "green"
    elif pct >= 50:
        color = "yellow"
    else:
        color = "red"
    badge = {
        "schemaVersion": 1,
        "label": "coverage",
        "message": msg,
        "color": color,
    }
    BADGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BADGE_PATH.write_text(json.dumps(badge, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {BADGE_PATH.relative_to(ROOT)} → {msg} ({color})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
