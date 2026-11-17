"""
Run every planning set in stages and rescore after each file, so results/summary.md
and figures/accuracy_by_complexity.png always hold the latest partial results.

Usage:
    python run_all.py --claude_bin C:/Users/<you>/.local/bin/claude.exe

Stops at the first file with failed calls (e.g. a usage limit); run it again later to resume.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
# (file, limit) in run order: all of Natural Plan and the large set first, then Blocksworld
# and Mystery Blocksworld in growing balanced samples.
STAGES = [
    ("natural_plan_trip", 100), ("natural_plan_meeting", 100), ("natural_plan_calendar", 100),
    ("blocksworld_hard", 110), ("blocksworld", 100), ("mystery_blocksworld", 100),
    ("blocksworld", 250), ("mystery_blocksworld", 250),
    ("blocksworld", 500), ("mystery_blocksworld", 500),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claude_bin", default="")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    for name, limit in STAGES:
        cmd = [sys.executable, str(HERE / "run_claude.py"), "--questions", str(HERE / "subsets" / f"{name}.jsonl"),
               "--limit", str(limit), "--workers", str(args.workers)]
        if args.claude_bin:
            cmd += ["--claude_bin", args.claude_bin]
        print(f"{time.strftime('%H:%M')} {name} up to {limit}", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        tail = [l for l in proc.stdout.splitlines() + proc.stderr.splitlines() if "failed" in l or "stopping" in l]
        subprocess.run([sys.executable, str(HERE / "score.py"), "--figure", str(HERE / "figures" / "accuracy_by_complexity.png")],
                       capture_output=True)
        if proc.returncode or tail:
            print("\n".join(tail[-5:]) or proc.stderr[-500:])
            sys.exit(f"stopped at {name}; rerun to resume")
    print("all done")


if __name__ == "__main__":
    main()
