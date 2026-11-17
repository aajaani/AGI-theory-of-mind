"""Paired analysis: original vs disguised ARC tasks, attempt 1 only (pass@1).

Usage (from the arc-agi-benchmarking folder):
    python3 analyse_results.py \
        --orig-tasks data/evaluation --orig-subs submissions/main/original \
        --disg-tasks disguised       --disg-subs submissions/main/disguised \
        --log disguised/disguise_log.txt --out analysis

Writes analysis/summary.txt and analysis/per_task.csv. Standard library only.
"""
import argparse
import ast
import csv
import json
import random
import re
from math import comb
from pathlib import Path


def load_version(task_dir, sub_dir):
    """Return {task_id: info}. A test pair counts as correct only if attempt 1 matches exactly."""
    out = {}
    for task_file in sorted(Path(task_dir).glob("*.json")):
        tid = task_file.stem
        truth = [p["output"] for p in json.loads(task_file.read_text())["test"]]
        sub_file = Path(sub_dir) / f"{tid}.json"
        subs = json.loads(sub_file.read_text()) if sub_file.exists() else []
        pairs, missing, usage = [], 0, []
        for i, expected in enumerate(truth):
            att = subs[i].get("attempt_1") if i < len(subs) else None
            if not att:
                missing += 1
                pairs.append(False)
                continue
            pairs.append(att.get("answer") == expected)
            md = att.get("metadata") or {}
            u = md.get("usage") or {}
            usage.append({
                "prompt": u.get("prompt_tokens", 0) or 0,
                "output": u.get("completion_tokens", 0) or 0,
                "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0,
                "turns": md.get("num_turns") or 1,
            })
        out[tid] = {"pairs": pairs, "score": sum(pairs) / len(pairs), "missing": missing, "usage": usage}
    return out


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def bootstrap(ids, fn, reps=10000, seed=0):
    """95% percentile interval for fn(sample of task ids), resampling tasks."""
    rng = random.Random(seed)
    vals = sorted(fn([rng.choice(ids) for _ in ids]) for _ in range(reps))
    return vals[int(0.025 * reps)], vals[int(0.975 * reps) - 1]


def parse_log(path):
    """{task_id: (flip, zero_remapped)} from lines like '00576224.json: rotate_180, colours {0: 3, ...}'."""
    info = {}
    if not path or not Path(path).exists():
        return info
    for line in Path(path).read_text().splitlines():
        m = re.match(r"(\S+)\.json: (\w+), colours (\{.*\})", line.strip())
        if m:
            mapping = ast.literal_eval(m.group(3))
            info[m.group(1)] = (m.group(2), mapping.get(0, 0) != 0)
    return info


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-tasks", required=True)
    ap.add_argument("--orig-subs", required=True)
    ap.add_argument("--disg-tasks", required=True)
    ap.add_argument("--disg-subs", required=True)
    ap.add_argument("--log")
    ap.add_argument("--out", default="analysis")
    a = ap.parse_args()

    orig = load_version(a.orig_tasks, a.orig_subs)
    disg = load_version(a.disg_tasks, a.disg_subs)
    ids = sorted(set(orig) & set(disg))
    log = parse_log(a.log)
    lines = []
    say = lines.append

    # --- completeness ---------------------------------------------------
    miss_o = sum(orig[t]["missing"] for t in ids)
    miss_d = sum(disg[t]["missing"] for t in ids)
    n_pairs = sum(len(orig[t]["pairs"]) for t in ids)
    say(f"Tasks: {len(ids)}   test pairs: {n_pairs}   (scored on attempt 1 only = pass@1)")
    say(f"Missing attempts (counted as wrong): original {miss_o}, disguised {miss_d}")
    if miss_o or miss_d:
        say("  WARNING: rerun the batches until these are 0 before reporting results.")
    turns = [u["turns"] for v in (orig, disg) for t in ids for u in v[t]["usage"]]
    say(f"Max turns in any attempt: {max(turns) if turns else 'n/a'} (1 = no tool use)")

    # --- accuracy (official ARC task score: share of a task's test pairs solved) ---
    acc = lambda v, s: 100 * mean([v[t]["score"] for t in s])
    gap = lambda s: acc(orig, s) - acc(disg, s)
    say("\nACCURACY (task-level, 95% bootstrap interval)")
    for name, v in (("Original", orig), ("Disguised", disg)):
        lo, hi = bootstrap(ids, lambda s, v=v: acc(v, s))
        say(f"  {name:<10} {acc(v, ids):5.1f}%   [{lo:.1f}, {hi:.1f}]")
    lo, hi = bootstrap(ids, gap)
    say(f"  Gap (orig - disguised): {gap(ids):+.1f} pts   [{lo:+.1f}, {hi:+.1f}]")

    # --- paired table and McNemar (unit = test pair) ----------------------
    both = only_o = only_d = neither = 0
    for t in ids:
        for po, pd in zip(orig[t]["pairs"], disg[t]["pairs"]):
            if po and pd: both += 1
            elif po: only_o += 1
            elif pd: only_d += 1
            else: neither += 1
    say("\nPAIRED TABLE (test pairs)")
    say(f"  Solved in both:            {both}")
    say(f"  Only original solved:      {only_o}")
    say(f"  Only disguised solved:     {only_d}")
    say(f"  Neither:                   {neither}")
    say(f"  Exact McNemar p-value:     {mcnemar_exact(only_o, only_d):.4f}")

    # --- tokens -------------------------------------------------------------
    say("\nTOKENS PER ATTEMPT (mean)")
    for name, v in (("Original", orig), ("Disguised", disg)):
        us = [u for t in ids for u in v[t]["usage"]]
        say(f"  {name:<10} prompt {mean([u['prompt'] for u in us]):8.0f}   "
            f"output {mean([u['output'] for u in us]):8.0f}   "
            f"reasoning {mean([u['reasoning'] for u in us]):8.0f}")

    # --- by disguise type ----------------------------------------------------
    if log:
        say("\nBY DISGUISE TYPE (task-level accuracy)")
        groups = {}
        for t in ids:
            if t in log:
                flip, zero = log[t]
                groups.setdefault(f"flip={flip}", []).append(t)
                groups.setdefault(f"background 0 {'remapped' if zero else 'kept'}", []).append(t)
        for g, s in sorted(groups.items()):
            say(f"  {g:<32} n={len(s):3d}   original {acc(orig, s):5.1f}%   "
                f"disguised {acc(disg, s):5.1f}%   gap {gap(s):+5.1f}")
        say("  (small groups: descriptive only)")

    say("\nReminder: ARC Prize's 87.5% for Opus 5.5 medium is pass@2 on a held-out set; "
        "pass@1 here should be somewhat lower even without contamination.")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    with open(out / "per_task.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "test_pairs", "orig_score", "disg_score", "flip", "zero_remapped"])
        for t in ids:
            flip, zero = log.get(t, ("", ""))
            w.writerow([t, len(orig[t]["pairs"]), orig[t]["score"], disg[t]["score"], flip, zero])
    print("\n".join(lines))
    print(f"\nSaved {out/'summary.txt'} and {out/'per_task.csv'}")


if __name__ == "__main__":
    main()
