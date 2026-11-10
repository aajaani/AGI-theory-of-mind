import argparse
import collections
import json
import math

from tombench_data import HUMAN_TASK_ACC, HUMAN_TASK_AVG, TASK_FILES, load_all

RUN_KEYS = ["backend", "model", "effort", "lang", "cot", "seed", "variant", "questions_file"]


def load_runs(paths):
    runs = collections.OrderedDict()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    row.setdefault("backend", "api")
                    runs.setdefault(tuple(row[k] for k in RUN_KEYS), []).append(row)
    return runs


def run_label(key):
    s = dict(zip(RUN_KEYS, key))
    return f"{s['backend']} {s['model']} effort={s['effort']} lang={s['lang']} variant={s['variant']}" + (" cot" if s["cot"] else "")


def vote(rows):
    # Majority vote over the shuffled attempts. Unparseable answers count as a
    # vote for "no answer"; a tie for first place is scored as incorrect.
    counts = collections.Counter(r["pred"] for r in rows)
    top = max(counts.values())
    winners = [p for p, c in counts.items() if c == top]
    pred = winners[0] if len(winners) == 1 else None
    return pred, len(winners) > 1


def score_questions(rows):
    by_q = collections.defaultdict(list)
    for r in rows:
        by_q[r["qid"]].append(r)
    attempts = collections.Counter(len(v) for v in by_q.values())
    questions = {}
    for qid, rs in by_q.items():
        pred, tie = vote(rs)
        meta = rs[0]
        questions[qid] = {
            "file": meta["file"],
            "is_task": meta["is_task"],
            "story_id": meta["story_id"],
            "ability": meta["ability"],
            "category": meta["category"],
            "answer": meta["answer"],
            "pred": pred,
            "tie": tie,
            "correct": pred == meta["answer"],
            "attempts": len(rs),
        }
    return questions, attempts


def accuracy(qs):
    return 100.0 * sum(q["correct"] for q in qs) / len(qs) if qs else float("nan")


def summarize(rows, task_sizes, story_members):
    questions, attempts = score_questions(rows)
    out = {"n_questions": len(questions), "attempts_per_question": dict(attempts)}

    tasks = {}
    var = 0.0
    for task in TASK_FILES:
        qs = [q for q in questions.values() if q["file"] == task]
        if not qs:
            continue
        n, N = len(qs), task_sizes[task]
        p = accuracy(qs) / 100
        fpc = (N - n) / (N - 1) if N > 1 else 0.0
        var += p * (1 - p) / n * fpc
        tasks[task] = {"n": n, "acc": round(100 * p, 2), "human": HUMAN_TASK_ACC[task]}
    out["tasks"] = tasks
    if tasks:
        avg = sum(t["acc"] for t in tasks.values()) / len(tasks)
        half = 1.96 * 100 * math.sqrt(var) / len(tasks)
        if len(tasks) < len(TASK_FILES):
            verdict = f"only {len(tasks)} of {len(TASK_FILES)} tasks present, not comparable to the human average"
        elif avg - half > HUMAN_TASK_AVG:
            verdict = "above the human average"
        elif avg + half < HUMAN_TASK_AVG:
            verdict = "below the human average"
        else:
            verdict = "inconclusive: the 95% interval includes the human average"
        out["task_average"] = {"acc": round(avg, 2), "ci95": round(half, 2), "human": HUMAN_TASK_AVG, "verdict": verdict}

    for field in ("category", "ability"):
        groups = collections.defaultdict(list)
        for q in questions.values():
            groups[q[field]].append(q)
        out[field] = {k: {"n": len(v), "acc": round(accuracy(v), 2)} for k, v in sorted(groups.items())}

    # Story-level coherent test: a story counts only if every one of its
    # questions is answered correctly. Only stories fully present are scored.
    coherent = {}
    for task in tasks:
        stories = [s for s, members in story_members.items() if s.startswith(task + "#")
                   and all(m in questions for m in members)]
        if stories:
            ok = sum(all(questions[m]["correct"] for m in story_members[s]) for s in stories)
            coherent[task] = {"stories": len(stories), "acc": round(100.0 * ok / len(stories), 2)}
    out["coherent"] = coherent

    statuses = collections.Counter(r["parse_status"] for r in rows)
    out["attempts"] = {
        "total": len(rows),
        "parse_status": dict(statuses),
        "unparsed_rate": round(100.0 * (len(rows) - statuses.get("ok", 0)) / len(rows), 2),
        "ties": sum(q["tie"] for q in questions.values()),
        "input_tokens": sum(r["usage"]["input_tokens"] for r in rows),
        "output_tokens": sum(r["usage"]["output_tokens"] for r in rows),
        "cost_usd": round(sum(r.get("cost_usd") or 0 for r in rows), 4),
    }
    return out, questions


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def compare(qa, qb):
    shared = sorted(set(qa) & set(qb))
    table = collections.Counter((qa[q]["correct"], qb[q]["correct"]) for q in shared)
    b, c = table[(True, False)], table[(False, True)]
    per_task = {}
    for task in TASK_FILES:
        ids = [q for q in shared if qa[q]["file"] == task]
        if ids:
            per_task[task] = {
                "n": len(ids),
                "acc_a": round(accuracy([qa[q] for q in ids]), 2),
                "acc_b": round(accuracy([qb[q] for q in ids]), 2),
            }
    return {
        "shared_questions": len(shared),
        "both_correct": table[(True, True)],
        "only_a_correct": b,
        "only_b_correct": c,
        "both_wrong": table[(False, False)],
        "acc_a": round(accuracy([qa[q] for q in shared]), 2),
        "acc_b": round(accuracy([qb[q] for q in shared]), 2),
        "mcnemar_p": mcnemar_exact(b, c),
        "tasks": per_task,
    }


def print_summary(label, s):
    print(f"\n=== {label} ===")
    print(f"{s['n_questions']} questions, attempts per question: {s['attempts_per_question']}")
    for task, t in s["tasks"].items():
        print(f"  {task:28s} n={t['n']:4d}  acc={t['acc']:6.2f}  (human {t['human']})")
    if "task_average" in s:
        a = s["task_average"]
        print(f"  {'TASK AVERAGE':28s}         acc={a['acc']:6.2f} ±{a['ci95']:.2f}  (human {a['human']}) -> {a['verdict']}")
    for cat, c in s["category"].items():
        print(f"  category {cat:30s} n={c['n']:4d}  acc={c['acc']:6.2f}")
    if s["coherent"]:
        print("  coherent (story-level), complete stories only:")
        for task, c in s["coherent"].items():
            print(f"    {task:26s} stories={c['stories']:4d}  acc={c['acc']:6.2f}")
    at = s["attempts"]
    print(f"  attempts={at['total']}  unparsed={at['unparsed_rate']}%  {at['parse_status']}  vote ties={at['ties']}")
    print(f"  tokens in={at['input_tokens']:,} out={at['output_tokens']:,}  cost ~${at['cost_usd']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score run_claude.py results against the ToMBench human baseline.")
    parser.add_argument("results", nargs="+", help="results files written by run_claude.py")
    parser.add_argument("--compare", action="store_true", help="paired comparison of exactly two runs (e.g. original vs swapped)")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    records = load_all()
    task_sizes = collections.Counter(r["file"] for r in records)
    story_members = collections.defaultdict(list)
    for r in records:
        story_members[r["story_id"]].append(r["qid"])

    runs = load_runs(args.results)
    report = {}
    scored = {}
    for key, rows in runs.items():
        label = run_label(key)
        summary, questions = summarize(rows, task_sizes, story_members)
        report[label] = summary
        scored[label] = questions
        print_summary(label, summary)

    if args.compare:
        if len(scored) != 2:
            raise SystemExit(f"--compare needs exactly two runs, found {len(scored)}")
        (la, qa), (lb, qb) = scored.items()
        cmp = compare(qa, qb)
        report["comparison"] = {"a": la, "b": lb, **cmp}
        print(f"\n=== paired comparison ===\nA: {la}\nB: {lb}")
        print(f"  {cmp['shared_questions']} shared questions: accuracy A={cmp['acc_a']}  B={cmp['acc_b']}")
        print(f"  both correct={cmp['both_correct']}  only A={cmp['only_a_correct']}  only B={cmp['only_b_correct']}  both wrong={cmp['both_wrong']}")
        print(f"  McNemar exact p={cmp['mcnemar_p']:.2g}")
        for task, t in cmp["tasks"].items():
            print(f"    {task:28s} n={t['n']:4d}  A={t['acc_a']:6.2f}  B={t['acc_b']:6.2f}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
