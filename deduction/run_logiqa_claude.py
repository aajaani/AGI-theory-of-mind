"""
Evaluate Claude (via Claude Code, using your Claude Pro login) on LogiQA 2.0 dev subset.

Usage (PowerShell, from the folder that contains this script):
    python run_logiqa_claude.py --data LogiQA2.0\\logiqa\\DATA\\LOGIQA\\dev.txt --n 100

Re-running the same command resumes where it stopped (e.g. after a usage limit).
Outputs:
    preds.jsonl   one line per question (prediction, gold, categories, raw answer)
    results.txt   accuracy per reasoning category + overall
"""
import argparse
import json
import random
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

LETTERS = "ABCD"

# The dataset's "type" field is a dict like {"Sufficient Conditional Reasoning": true, ...}.
# Key spellings in the data have typos in places, so match on a keyword instead of the exact key.
CATEGORIES = [
    ("Categorical reasoning", "categorical"),
    ("Sufficient conditional reasoning", "sufficient"),
    ("Necessary conditional reasoning", "necess"),
    ("Disjunctive reasoning", "disjunctive"),
    ("Conjunctive reasoning", "conjunctive"),
]

SYSTEM_PROMPT = (
    "You are taking a logical reasoning test. Read the passage and question carefully, "
    "reason step by step, and finish with a final line of the form 'Answer: X' "
    "where X is one of A, B, C, D."
)


def categories_of(example):
    t = example.get("type") or {}
    if isinstance(t, list):  # be tolerant of a list-of-names format
        keys = [str(k).lower() for k in t]
    else:
        keys = [str(k).lower() for k, v in t.items() if v]
    return [name for name, kw in CATEGORIES if any(kw in k for k in keys)]


def build_prompt(ex):
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(ex["options"]))
    return (
        f"Passage:\n{ex['text']}\n\n"
        f"Question: {ex['question']}\n\n"
        f"Options:\n{opts}\n\n"
        "Think step by step, then end with 'Answer: X' (A, B, C or D)."
    )


def parse_answer(text):
    m = re.findall(r"Answer\s*[:：]\s*\**\s*\(?([ABCD])\b", text, flags=re.IGNORECASE)
    return LETTERS.index(m[-1].upper()) if m else -1


def ask_claude(claude_bin, prompt, model, effort, workdir, system_prompt=SYSTEM_PROMPT):
    cmd = [
        claude_bin, "-p",
        "--model", model,
        "--output-format", "json",
        "--tools", "",                     # no tools: answer from the prompt only
        "--system-prompt", system_prompt,  # replace Claude Code's coding-agent prompt
        "--no-session-persistence",
        "--safe-mode",                     # ignore CLAUDE.md, plugins, hooks, MCP servers
    ]
    if effort:
        cmd += ["--effort", effort]
    r = subprocess.run(
        cmd, input=prompt, capture_output=True, text=True,
        encoding="utf-8", errors="replace", cwd=workdir, timeout=900,
    )
    try:
        out = json.loads(r.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"Claude Code failed (exit {r.returncode}):\n{r.stdout}\n{r.stderr}")
    if out.get("is_error"):
        raise RuntimeError(f"Claude Code returned an error: {out.get('result')}")
    u = out.get("usage") or {}
    meta = {
        "input_tokens": u.get("input_tokens", 0) or 0,
        "output_tokens": u.get("output_tokens", 0) or 0,
        "cache_read_tokens": u.get("cache_read_input_tokens", 0) or 0,
        "cache_write_tokens": u.get("cache_creation_input_tokens", 0) or 0,
        "cost_usd": out.get("total_cost_usd", 0) or 0,
        "duration_s": round((out.get("duration_ms", 0) or 0) / 1000, 1),
    }
    # Keep any rate-limit / quota info Claude Code chooses to report (field names vary by version).
    limits = {k: v for k, v in out.items() if "limit" in k.lower() or "quota" in k.lower()}
    if limits:
        meta["limits"] = limits
    return out.get("result", ""), meta


def usage_summary(preds):
    rows = [p.get("usage") for p in preds if p.get("usage")]
    if not rows:
        return ""
    tot = lambda k: sum(r.get(k, 0) for r in rows)
    n = len(rows)
    lines = [
        "",
        f"Usage over {n} questions:",
        f"  Input tokens        {tot('input_tokens'):>10,}",
        f"  Output tokens       {tot('output_tokens'):>10,}   (avg {tot('output_tokens') / n:,.0f} per question)",
        f"  Cache read tokens   {tot('cache_read_tokens'):>10,}",
        f"  Cache write tokens  {tot('cache_write_tokens'):>10,}",
        f"  API-equivalent cost ${tot('cost_usd'):>9.2f}   (avg ${tot('cost_usd') / n:.3f} per question;"
        " not charged on Pro)",
        f"  Time                {tot('duration_s') / 60:>9.1f} min",
    ]
    last_limits = next((r["limits"] for r in reversed(rows) if r.get("limits")), None)
    if last_limits:
        lines.append(f"  Limit info reported by Claude Code: {json.dumps(last_limits)}")
    lines.append("  Remaining Pro allowance: run /usage in an interactive `claude` session,"
                 " or see claude.ai > Settings > Usage.")
    return "\n".join(lines)


def write_results(preds, path):
    lines = []
    header = f"{'Category':36s} {'Correct':>8s} {'Total':>6s} {'Accuracy':>9s}"
    lines += [header, "-" * len(header)]
    for name, _ in CATEGORIES:
        rows = [p for p in preds if name in p["categories"]]
        c = sum(p["correct"] for p in rows)
        acc = f"{100 * c / len(rows):.1f}%" if rows else "n/a"
        lines.append(f"{name:36s} {c:8d} {len(rows):6d} {acc:>9s}")
    lines.append("-" * len(header))
    c = sum(p["correct"] for p in preds)
    lines.append(f"{'Overall (all questions)':36s} {c:8d} {len(preds):6d} "
                 f"{100 * c / len(preds):8.1f}%")
    per_cat = [
        sum(p["correct"] for p in preds if name in p["categories"])
        / max(1, sum(name in p["categories"] for p in preds))
        for name, _ in CATEGORIES
        if any(name in p["categories"] for p in preds)
    ]
    if per_cat:
        lines.append(f"{'Macro average of categories':36s} {'':8s} {'':6s} "
                     f"{100 * sum(per_cat) / len(per_cat):8.1f}%")
    unparsed = sum(p["pred"] == -1 for p in preds)
    lines.append(f"\nUnparseable answers (counted wrong): {unparsed}")
    lines.append("Note: a question can belong to several categories, so category totals "
                 "add up to more than the number of questions.")
    text = "\n".join(lines)
    Path(path).write_text(text + "\n", encoding="utf-8")
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="LogiQA2.0/logiqa/DATA/LOGIQA/dev.txt")
    ap.add_argument("--n", type=int, default=100, help="number of dev questions")
    ap.add_argument("--seed", type=int, default=42, help="random sample seed")
    ap.add_argument("--model", default="opus", help="opus, sonnet, haiku or a full model name")
    ap.add_argument("--effort", default="", help="optional: low, medium, high, xhigh, max")
    ap.add_argument("--out", default="preds.jsonl")
    ap.add_argument("--results", default="results.txt")
    args = ap.parse_args()

    claude_bin = shutil.which("claude")
    if not claude_bin:
        sys.exit("Could not find 'claude' on PATH. Open a new terminal or fix PATH first.")

    data = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    if not all("variant" in ex for ex in data):
        # original LogiQA file: shuffle with the seed, then take the first n
        random.Random(args.seed).shuffle(data)
    # variant files from make_variants.py are already in that same order, so don't reshuffle
    subset = data[: args.n]

    out_path = Path(args.out)
    done = {}
    if out_path.exists():
        for l in out_path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                p = json.loads(l)
                done[p["id"]] = p

    workdir = tempfile.mkdtemp(prefix="logiqa_empty_")  # empty folder, nothing to peek at
    todo = [ex for ex in subset if ex["id"] not in done]
    print(f"{len(subset)} questions selected, {len(done)} already done, {len(todo)} to run.")

    with out_path.open("a", encoding="utf-8") as f:
        for i, ex in enumerate(todo, 1):
            try:
                raw, meta = ask_claude(claude_bin, build_prompt(ex), args.model, args.effort, workdir)
            except Exception as e:
                print(f"\nStopped at question {ex['id']}: {e}\n"
                      "Progress is saved; run the same command again later to resume.")
                break
            pred = parse_answer(raw)
            rec = {
                "id": ex["id"],
                "gold": ex["answer"],
                "pred": pred,
                "gold_letter": LETTERS[ex["answer"]],
                "pred_letter": LETTERS[pred] if pred >= 0 else None,
                "correct": pred == ex["answer"],
                "categories": categories_of(ex),
                "text": ex["text"],
                "question": ex["question"],
                "options": ex["options"],
                "usage": meta,
            }
            for k in ("variant", "perm"):  # carried over from variant files (make_variants.py)
                if k in ex:
                    rec[k] = ex[k]
            if pred == -1:  # keep the model's reply only when no answer could be parsed
                rec["raw"] = raw
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            done[ex["id"]] = rec
            mark = "OK " if rec["correct"] else "X  "
            print(f"[{len(done):3d}/{len(subset)}] {mark} id={ex['id']} "
                  f"pred={LETTERS[pred] if pred >= 0 else '?'} gold={LETTERS[ex['answer']]}  "
                  f"out_tok={meta['output_tokens']:,} cost~${meta['cost_usd']:.3f} "
                  f"{meta['duration_s']}s")

    preds = [done[ex["id"]] for ex in subset if ex["id"] in done]
    if preds:
        report = write_results(preds, args.results) + usage_summary(preds)
        Path(args.results).write_text(report + "\n", encoding="utf-8")
        print("\n" + report)
        if len(preds) < len(subset):
            print(f"\n(Partial: {len(preds)}/{len(subset)} questions so far.)")


if __name__ == "__main__":
    main()