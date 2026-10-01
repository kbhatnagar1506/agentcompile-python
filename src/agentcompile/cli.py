"""`agentcompile trail`: see what AgentCompile did with each model call."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from . import __version__
from ._trail import DEFAULT_PATH

COLORS = {
    "compiled": "\033[33m",
    "forwarded": "\033[0m",
    "fail-open": "\033[31m",
    "shadow": "\033[36m",
    "no-conversation": "\033[2m",
}
RESET = "\033[0m"


def line(event: dict[str, Any], color: bool) -> str:
    ts = datetime.fromtimestamp(event.get("ts", 0)).strftime("%H:%M:%S")
    route = event.get("route", "?")
    what = event.get("tool") or event.get("action") or event.get("reason") or ""
    if route in ("forwarded", "fail-open") and event.get("reason"):
        what = event["reason"]
    timing = f"{event['decide_ms']:.0f} ms decide" if event.get("decide_ms") is not None else ""
    convo = str(event.get("conversation", "-"))[:12]
    model = event.get("model", "")
    text = f"{ts}  {convo:<12}  {route:<15}  {str(what)[:40]:<40}  {model}  {timing}"
    return f"{COLORS.get(route, '')}{text}{RESET}" if color else text


def follow(path: Path, start_at_end: bool) -> Iterator[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    with path.open(encoding="utf-8") as fh:
        if start_at_end:
            fh.seek(0, 2)
        while True:
            row = fh.readline()
            if row:
                yield row
            else:
                time.sleep(0.25)


def summary(rows: list) -> str:
    total = len(rows)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.get("route", "?")] = counts.get(r.get("route", "?"), 0) + 1
    parts = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
    compiled = counts.get("compiled", 0)
    share = f"{100 * compiled / total:.0f}%" if total else "0%"
    return f"{total} calls: {parts}. Model calls avoided: {compiled} ({share})."


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(prog="agentcompile")
    parser.add_argument("--version", action="version", version=f"agentcompile {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("trail", help="show the trail of model calls")
    t.add_argument("-f", "--follow", action="store_true", help="keep printing new calls")
    t.add_argument("-n", type=int, default=50, help="how many recent calls to show")
    t.add_argument("--path", type=Path, default=DEFAULT_PATH)
    t.add_argument("--json", action="store_true", help="raw JSON lines")
    args = parser.parse_args(argv)
    color = sys.stdout.isatty() and not args.json
    rows = []
    if args.path.exists():
        for raw in args.path.read_text(encoding="utf-8").splitlines()[-args.n :]:
            try:
                rows.append(json.loads(raw))
            except ValueError:
                continue
    for r in rows:
        print(json.dumps(r) if args.json else line(r, color))
    if not args.follow:
        if not args.json:
            print(summary(rows))
        return 0
    try:
        for raw in follow(args.path, start_at_end=True):
            try:
                event = json.loads(raw)
            except ValueError:
                continue
            print(json.dumps(event) if args.json else line(event, color), flush=True)
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
