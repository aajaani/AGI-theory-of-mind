"""
Memorization probes for LogiQA 2.0 (uses variants/probes.jsonl from make_variants.py).

Two probe types, each asked in two ways:
  completion: give the first half of a passage, ask for the rest
  options:    give passage + question, ask for the four answer options
  plain:  no mention of the dataset
  guided: tells Claude the text is from LogiQA 2.0 and asks for it exactly as in the dataset

How to read it: a model that has NOT memorized the items writes a plausible but different
text, so overlap stays low in both modes. Clearly higher overlap in "guided" mode, or
near-verbatim matches (high ROUGE-L, long shared word sequences), suggest memorization.

Usage:
    python run_probes.py --probes variants\\probes.jsonl --out probes_results.jsonl
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from run_logiqa_claude import ask_claude

SYSTEM = "Follow the user's instruction exactly. Output only what is asked for, with no commentary."

PROMPTS = {
    ("completion", "plain"):
        "Continue the following text. Output only the continuation (about {n} words).\n\n{prefix}",
    ("completion", "guided"):
        "The following is the beginning of a passage from the test split of the LogiQA 2.0 "
        "dataset (logical reasoning questions translated from Chinese civil service exams). "
        "Complete the passage exactly as it appears in the dataset. Output only the "
        "continuation (about {n} words).\n\n{prefix}",
    ("options", "plain"):
        "Here is a logical reasoning question without its answer options. Write four answer "
        "options A-D for it, one per line, starting with 'A.', 'B.', 'C.', 'D.'.\n\n"
        "Passage: {text}\n\nQuestion: {question}",
    ("options", "guided"):
        "Here is a question from the test split of the LogiQA 2.0 dataset with its answer "
        "options removed. Write the four options A-D exactly as they appear in the dataset, "
        "one per line, starting with 'A.', 'B.', 'C.', 'D.'.\n\n"
        "Passage: {text}\n\nQuestion: {question}",
}


def toks(s):
    return [w.strip(".,;:!?\"'()[]").lower() for w in s.split() if w.strip(".,;:!?\"'()[]")]


def lcs(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l(pred, ref):
    p, r = toks(pred), toks(ref)
    if not p or not r:
        return 0.0
    l = lcs(p, r)
    if l == 0:
        return 0.0
    prec, rec = l / len(p), l / len(r)
    return 2 * prec * rec / (prec + rec)


def longest_shared_run(pred, ref):
    """Longest sequence of consecutive words that appears in both texts."""
    p, r = toks(pred), toks(ref)
    best = 0
    prev = [0] * (len(r) + 1)
    for x in p:
        cur = [0] * (len(r) + 1)
        for j, y in enumerate(r):
            if x == y:
                cur[j + 1] = prev[j] + 1
                best = max(best, cur[j + 1])
        prev = cur
    return best


def parse_options(reply):
    opts = {}
    for line in reply.splitlines():
        line = line.strip()
        if len(line) > 2 and line[0] in "ABCD" and line[1] in ".):":
            opts[line[0]] = line[2:].strip()
    return [opts.get(l, "") for l in "ABCD"]


def score(probe, reply):
    if probe["probe"] == "completion":
        ref = probe["target"]
        pred = " ".join(reply.split()[: len(ref.split()) + 10])
        return {"rougeL": rouge_l(pred, ref), "shared_run": longest_shared_run(pred, ref)}
    got = parse_options(reply)
    refs = probe["target_options"]
    # options can come back in another order: match each real option to its best guess
    per = [max(rouge_l(g, r) for g in got) if any(got) else 0.0 for r in refs]
    runs = [max(longest_shared_run(g, r) for g in got) if any(got) else 0 for r in refs]
    return {"rougeL": sum(per) / len(per), "shared_run": max(runs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", default="variants/probes.jsonl")
    ap.add_argument("--out", default="probes_results.jsonl")
    ap.add_argument("--model", default="opus")
    ap.add_argument("--effort", default="")
    ap.add_argument("--modes", default="plain,guided")
    ap.add_argument("--kinds", default="completion,options")
    ap.add_argument("--n", type=int, default=0,
                    help="only the first n questions (same order as the baseline); 0 = all")
    args = ap.parse_args()

    claude_bin = shutil.which("claude") or sys.exit("'claude' not found on PATH")
    probes = [json.loads(l) for l in open(args.probes, encoding="utf-8") if l.strip()]
    order = list(dict.fromkeys(p["id"] for p in probes))  # question ids in file order
    if args.n:
        if args.n > len(order):
            print(f"Note: probes file has only {len(order)} questions; "
                  f"re-run make_variants.py with --n {args.n} for more.")
        keep = set(order[: args.n])
        probes = [p for p in probes if p["id"] in keep]
    selected = {p["id"] for p in probes}
    print(f"Using {len(selected)} questions.")
    kinds, modes = args.kinds.split(","), args.modes.split(",")
    jobs = [(p, m) for p in probes if p["probe"] in kinds for m in modes]

    out_path = Path(args.out)
    done = {}
    if out_path.exists():
        for l in out_path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                done[(r["id"], r["probe"], r["mode"])] = r
    workdir = tempfile.mkdtemp(prefix="logiqa_empty_")
    todo = [(p, m) for p, m in jobs if (p["id"], p["probe"], m) not in done]
    print(f"{len(jobs)} probe calls, {len(jobs) - len(todo)} done, {len(todo)} to run.")

    with out_path.open("a", encoding="utf-8") as f:
        for p, m in todo:
            if p["probe"] == "completion":
                prompt = PROMPTS[("completion", m)].format(
                    prefix=p["prefix"], n=len(p["target"].split()))
            else:
                prompt = PROMPTS[("options", m)].format(text=p["text"], question=p["question"])
            try:
                reply, meta = ask_claude(claude_bin, prompt, args.model, args.effort, workdir,
                                         system_prompt=SYSTEM)
            except Exception as e:
                print(f"\nStopped: {e}\nRun the same command again later to resume.")
                break
            rec = {"id": p["id"], "probe": p["probe"], "mode": m, **score(p, reply),
                   "reply": reply, "usage": meta}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            done[(p["id"], p["probe"], m)] = rec
            print(f"{p['probe']:10s} {m:6s} id={p['id']}  rougeL={rec['rougeL']:.2f} "
                  f"shared_run={rec['shared_run']}")

    rows = [r for r in done.values() if r["id"] in selected]  # summary covers only this n
    if not rows:
        return
    print(f"\n{'Probe':12s} {'Mode':7s} {'n':>4s} {'mean ROUGE-L':>13s} "
          f"{'ROUGE-L>0.5':>12s} {'run>=8 words':>13s}")
    for k in kinds:
        for m in modes:
            rs = [r for r in rows if r["probe"] == k and r["mode"] == m]
            if not rs:
                continue
            mean = sum(r["rougeL"] for r in rs) / len(rs)
            hi = sum(r["rougeL"] > 0.5 for r in rs)
            run8 = sum(r["shared_run"] >= 8 for r in rs)
            print(f"{k:12s} {m:7s} {len(rs):4d} {mean:13.3f} {hi:12d} {run8:13d}")
    print("\nFlagged items (ROUGE-L > 0.5 or a shared run of 8+ words) are worth reading by hand.")


if __name__ == "__main__":
    main()