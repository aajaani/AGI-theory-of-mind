"""Compare the model with the ARC-AGI-2 human testing data (arcprize/arc_agi_2_human_testing).

Usage (from the arc-agi-benchmarking folder):
    python3 human_compare.py --human-csv arc_agi_2_human_testing.csv \
        --tasks data/evaluation --subs submissions/main/original

Only "Public Eval" rows are used, and only test pairs the model also answered.
The model is scored on attempt 1 (pass@1). A human row counts as solved by rule:
    pass1: correct on the first submission    (submissions == 1, correct >= 1)
    pass2: correct within two submissions     (submissions <= 2, correct >= 1)  <- ARC's official rule
    any:   correct at some point              (correct >= 1)
pass1/pass2 assume participants stop submitting once they are correct.
Standard library only.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

RULES = {
    "pass1": lambda s, c: c >= 1 and s == 1,
    "pass2": lambda s, c: c >= 1 and s <= 2,
    "any": lambda s, c: c >= 1,
}


def model_results(task_dir, sub_dir):
    """{(task_id, test_index): True/False} from attempt 1."""
    res = {}
    for f in sorted(Path(task_dir).glob("*.json")):
        truth = [p["output"] for p in json.loads(f.read_text())["test"]]
        sf = Path(sub_dir) / f.name
        subs = json.loads(sf.read_text()) if sf.exists() else []
        for i, expected in enumerate(truth):
            att = subs[i].get("attempt_1") if i < len(subs) else None
            res[(f.stem, i)] = bool(att) and att.get("answer") == expected
    return res


def load_humans(path, rule):
    """{session: {(task, test): solved}}, Public Eval only; best result if a pair repeats in a session."""
    solved = RULES[rule]
    sessions = defaultdict(dict)
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["task_set"].strip() != "Public Eval":
                continue
            key = (row["task_ID"].strip(), int(float(row["test_index"])))
            ok = solved(int(float(row["submissions"])), int(float(row["correct_submissions"])))
            sess = sessions[row["session_ID"].strip()]
            sess[key] = sess.get(key, False) or ok
    return sessions


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def analyse(sessions, model, min_items):
    sessions = {s: {k: v for k, v in d.items() if k in model} for s, d in sessions.items()}
    sessions = {s: d for s, d in sessions.items() if d}

    # Average human: per test pair solve rate, averaged over pairs with human data.
    by_pair = defaultdict(list)
    for d in sessions.values():
        for k, v in d.items():
            by_pair[k].append(v)
    pairs = sorted(by_pair)
    human_avg = 100 * mean([mean(by_pair[k]) for k in pairs])
    model_avg = 100 * mean([model[k] for k in pairs])

    # Percentile: each participant vs the model on exactly the same test pairs.
    scores, h_accs = [], []
    for d in sessions.values():
        if len(d) < min_items:
            continue
        h = mean(list(d.values()))
        m = mean([model[k] for k in d])
        h_accs.append(h)
        scores.append(1.0 if m > h else 0.5 if m == h else 0.0)
    pct = 100 * mean(scores)
    return {
        "pairs": len(pairs), "rows": sum(len(v) for v in by_pair.values()),
        "human_avg": human_avg, "model_avg": model_avg,
        "sessions": len(scores), "percentile": pct,
        "perfect_humans": 100 * mean([h == 1.0 for h in h_accs]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human-csv", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--subs", required=True)
    ap.add_argument("--min-items", type=int, default=3,
                    help="minimum eval test pairs a participant must have attempted (default 3)")
    ap.add_argument("--rule", choices=list(RULES), default="pass2",
                    help="rule used for the checkpoint verdict (default pass2, ARC's official rule)")
    a = ap.parse_args()

    model = model_results(a.tasks, a.subs)
    print(f"Model: {sum(model.values())}/{len(model)} test pairs correct on attempt 1 "
          f"({100 * mean(list(model.values())):.1f}%)\n")
    print(f"{'Human rule':<11}{'pairs':>6}{'rows':>7}{'human avg':>11}{'model, same pairs':>19}"
          f"{'participants':>14}{'model pctile':>14}{'humans at 100%':>16}")
    results = {}
    for rule in RULES:
        r = analyse(load_humans(a.human_csv, rule), model, a.min_items)
        results[rule] = r
        print(f"{rule:<11}{r['pairs']:>6}{r['rows']:>7}{r['human_avg']:>10.1f}%{r['model_avg']:>18.1f}%"
              f"{r['sessions']:>14}{r['percentile']:>13.1f}%{r['perfect_humans']:>15.1f}%")

    r = results[a.rule]
    above_avg = r["model_avg"] > r["human_avg"]
    if not above_avg:
        verdict = "below the human average -> no increase"
    elif r["percentile"] < 90:
        verdict = "above average, below the 90th percentile -> +20%"
    else:
        verdict = "at or above the 90th percentile -> +40%"
    print(f"\nCheckpoint rule (human rule = {a.rule}, participants with >= {a.min_items} eval pairs):")
    print(f"  Model {r['model_avg']:.1f}% vs human average {r['human_avg']:.1f}% on the same pairs; "
          f"model beats {r['percentile']:.1f}% of participants (ties count half).")
    print(f"  Verdict: {verdict}")
    print("\nNotes: each participant saw only a few tasks, so percentiles compare the model with each "
          "person on that person's own tasks.\nIf many humans score 100% on their tasks, the model can tie "
          "but never beat them, which caps its percentile.")


if __name__ == "__main__":
    main()
