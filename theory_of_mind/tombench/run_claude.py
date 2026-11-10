import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from prompts import *
from tombench_data import LETTERS, read_questions

ANSWER_RE = re.compile(r"\[\[\s*([A-D])\s*\]\]")

# Settings that must match for rows to be appended to an existing results file.
RUN_KEYS = ["backend", "model", "effort", "lang", "cot", "seed", "variant", "questions_file"]

# USD per million input / output tokens, for the spending cap and reports.
PRICES = {"claude-opus-5-5": (4.0, 20.0)}

DEFAULT_KEY_FILE = os.path.expanduser("~/.config/tombench/anthropic_api_key")

# Removed from the environment of `claude -p`, so it runs on the logged-in
# subscription and not on API credits, and does not detect a parent session.
CLAUDE_CODE_DROP_ENV = {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT"}


class CallFailed(Exception):
    pass


class BudgetReached(Exception):
    pass


def permutation(seed, qid, attempt, n):
    # Seeded by question id, not by option text, so the original and the
    # name/object-swapped variant of a question get identical shuffles.
    perm = list(range(n))
    random.Random(f"{seed}|{qid}|{attempt}").shuffle(perm)
    return perm


def build_prompt(q, lang, perm, cot):
    item = q[lang]
    shown = [item["options"][i] for i in perm]
    if lang == "zh":
        system = SystemEvaluatePrompt_zh_cot if cot else SystemEvaluatePrompt_zh
        template = UserEvaluatePrompt4Choices_zh if len(shown) == 4 else UserEvaluatePrompt2Choices_zh
    else:
        system = SystemEvaluatePrompt_en_cot if cot else SystemEvaluatePrompt_en
        template = UserEvaluatePrompt4Choices_en if len(shown) == 4 else UserEvaluatePrompt2Choices_en
    fields = {f"choice_{l.lower()}": o for l, o in zip(LETTERS, shown)}
    return system, template.format(story=item["story"], question=item["question"], **fields)


def parse_answer(text, n):
    found = set(ANSWER_RE.findall(text or ""))
    if not found:
        return None, "no_answer"
    if len(found) > 1:
        return None, "multiple_answers"
    letter = found.pop()
    if LETTERS.index(letter) >= n:
        return None, "out_of_range"
    return letter, "ok"


def load_done(path, settings):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            for k in RUN_KEYS:
                if row.get(k, "api" if k == "backend" else None) != settings[k]:
                    sys.exit(f"{path} holds results with {k}={row.get(k)!r}, this run has {settings[k]!r}; use a different --output")
            done.add((row["qid"], row["attempt"]))
    return done


class Runner:
    def __init__(self, args, settings):
        self.args = args
        self.settings = settings
        self.lock = threading.Lock()
        self.spent = 0.0
        if args.backend == "api":
            import anthropic

            self.anthropic = anthropic
            key = None
            if os.path.exists(args.api_key_file):
                with open(args.api_key_file) as f:
                    key = f.read().strip()
            self.client = anthropic.Anthropic(api_key=key, max_retries=args.max_retries)
        else:
            self.workdir = tempfile.mkdtemp(prefix="tombench_claude_code_")
            self.env = {k: v for k, v in os.environ.items() if k not in CLAUDE_CODE_DROP_ENV}
            # Claude Code otherwise enables a server-side advisor tool (a second
            # model the answering model can consult), even with --tools "".
            self.env["CLAUDE_CODE_DISABLE_ADVISOR_TOOL"] = "1"

    def call_api(self, system, prompt):
        response = self.client.messages.create(
            model=self.args.model,
            max_tokens=self.args.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": self.args.effort},
        )
        usage = response.usage
        price = PRICES.get(self.args.model)
        return {
            "text": "".join(b.text for b in response.content if b.type == "text"),
            "stop_reason": response.stop_reason,
            "stop_details": response.stop_details.to_dict() if response.stop_details else None,
            "usage": {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens},
            "cost_usd": usage.input_tokens * price[0] / 1e6 + usage.output_tokens * price[1] / 1e6 if price else None,
            "response_model": response.model,
            "response_id": response.id,
            "request_id": response._request_id,
        }

    def call_claude_code(self, system, prompt):
        # Headless Claude Code in an empty directory with every tool disabled and
        # the benchmark prompt replacing Claude Code's system prompt.
        cmd = [self.args.claude_bin, "-p", prompt, "--output-format", "json",
               "--model", self.args.model, "--effort", self.args.effort,
               "--system-prompt", system, "--tools", "", "--strict-mcp-config",
               "--no-session-persistence"]
        try:
            proc = subprocess.run(cmd, cwd=self.workdir, env=self.env, capture_output=True,
                                  text=True, timeout=self.args.timeout)
        except subprocess.TimeoutExpired:
            raise CallFailed(f"claude -p timed out after {self.args.timeout}s")
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            raise CallFailed(f"claude -p exit {proc.returncode}: {(proc.stderr or proc.stdout)[:300]}")
        if data.get("is_error") or data.get("subtype") != "success":
            raise CallFailed(f"claude -p error: {data.get('subtype')} {str(data.get('result'))[:300]}")
        usage = data.get("usage") or {}
        models = list((data.get("modelUsage") or {}).keys())
        return {
            "text": data.get("result") or "",
            "stop_reason": data.get("stop_reason"),
            "stop_details": None,
            "usage": {
                "input_tokens": usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0)
                + usage.get("cache_read_input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "thinking_tokens": (usage.get("output_tokens_details") or {}).get("thinking_tokens"),
            },
            # List-price equivalent reported by Claude Code; not billed on a subscription.
            "cost_usd": data.get("total_cost_usd"),
            "response_model": models[0] if len(models) == 1 else models,
            "response_id": data.get("session_id"),
            "request_id": data.get("uuid"),
            # Evidence that no tools ran: one turn, no web requests, no blocked tool calls.
            "num_turns": data.get("num_turns"),
            "web_requests": sum((usage.get("server_tool_use") or {}).values()),
            "permission_denials": len(data.get("permission_denials") or []),
            "iteration_types": [it.get("type") for it in usage.get("iterations") or []],
            "advisor_disabled": True,
        }

    def ask(self, q, attempt):
        with self.lock:
            if self.args.max_cost and self.spent >= self.args.max_cost:
                raise BudgetReached()
        n = len(q[self.args.lang]["options"])
        perm = permutation(self.args.seed, q["qid"], attempt, n)
        system, prompt = build_prompt(q, self.args.lang, perm, self.args.cot)

        result = self.call_api(system, prompt) if self.args.backend == "api" else self.call_claude_code(system, prompt)

        if result["stop_reason"] in ("refusal", "max_tokens"):
            shown, status = None, result["stop_reason"]
        else:
            shown, status = parse_answer(result["text"], n)
        pred = LETTERS[perm[LETTERS.index(shown)]] if shown else None

        row = dict(self.settings)
        row.update({
            "qid": q["qid"],
            "attempt": attempt,
            "file": q["file"],
            "is_task": q["is_task"],
            "story_id": q["story_id"],
            "ability": q["ability"],
            "category": q["category"],
            "answer": q["answer"],
            "perm": perm,
            "shown_pred": shown,
            "pred": pred,
            "correct": pred == q["answer"],
            "parse_status": status,
            **result,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        with self.lock:
            self.spent += result["cost_usd"] or 0.0
            with open(self.args.output, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row


def dry_run(questions, args):
    for q in questions[: args.limit or 2]:
        n = len(q[args.lang]["options"])
        perm = permutation(args.seed, q["qid"], 0, n)
        system, prompt = build_prompt(q, args.lang, perm, args.cot)
        shown_answer = LETTERS[perm.index(LETTERS.index(q["answer"]))]
        print(f"=== {q['qid']} ({q['variant']}) attempt 0, correct shown as {shown_answer} ===")
        print(f"[system]\n{system}\n[user]\n{prompt}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ToMBench questions against a Claude model.")
    parser.add_argument("--questions", type=str, required=True, help="question file in subsets/ (original or swapped)")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--backend", type=str, default="api", choices=["api", "claude-code"],
                        help="api: Messages API (API credits); claude-code: `claude -p` (Claude subscription)")
    parser.add_argument("--model", type=str, default="claude-opus-5-5")
    parser.add_argument("--effort", type=str, default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--language", dest="lang", type=str, default="en", choices=["en", "zh"])
    parser.add_argument("--cot", action="store_true", help="use the paper's CoT system prompt instead of the vanilla one")
    parser.add_argument("--try_times", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=16000, help="api backend only")
    parser.add_argument("--max_cost", type=float, default=0.0, help="stop starting new calls once this many USD are spent (0 = no cap)")
    parser.add_argument("--api_key_file", type=str, default=DEFAULT_KEY_FILE, help="file holding the API key; falls back to ANTHROPIC_API_KEY")
    parser.add_argument("--claude_bin", type=str, default="claude")
    parser.add_argument("--timeout", type=int, default=600, help="claude-code backend: seconds per call")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max_retries", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="only the first N questions (for test runs)")
    parser.add_argument("--dry_run", action="store_true", help="print prompts without calling the model")
    args = parser.parse_args()

    questions = read_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]
    variants = {q.get("variant", "original") for q in questions}
    if len(variants) != 1:
        sys.exit(f"question file mixes variants {variants}")
    for q in questions:
        q.setdefault("variant", "original")
    if any(q.get("swap_lang", args.lang) != args.lang for q in questions):
        sys.exit(f"this question file was swapped in {questions[0]['swap_lang']!r} only; run it with --language {questions[0]['swap_lang']}")

    if args.dry_run:
        dry_run(questions, args)
        sys.exit(0)

    settings = {
        "backend": args.backend,
        "model": args.model,
        "effort": args.effort,
        "lang": args.lang,
        "cot": args.cot,
        "seed": args.seed,
        "variant": variants.pop(),
        "questions_file": os.path.basename(args.questions),
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    done = load_done(args.output, settings)
    todo = [(q, j) for q in questions for j in range(args.try_times) if (q["qid"], j) not in done]
    print(f"{len(questions)} questions x {args.try_times} attempts: {len(done)} already done, {len(todo)} to run ({args.backend})")

    runner = Runner(args, settings)
    failed = skipped = 0
    statuses = {}
    fatal = ()
    if args.backend == "api":
        a = runner.anthropic
        fatal = (a.BadRequestError, a.AuthenticationError, a.PermissionDeniedError, a.NotFoundError)
        transient = (a.APIStatusError, a.APIConnectionError, CallFailed)
    else:
        transient = (CallFailed,)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(runner.ask, q, j): (q["qid"], j) for q, j in todo}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            try:
                row = fut.result()
                statuses[row["parse_status"]] = statuses.get(row["parse_status"], 0) + 1
            except BudgetReached:
                skipped += 1
            except fatal as e:
                pool.shutdown(wait=False, cancel_futures=True)
                sys.exit(f"stopping: {type(e).__name__}: {e}")
            except transient as e:
                failed += 1
                tqdm.write(f"{futures[fut]} failed: {type(e).__name__}: {e}")

    print(f"done. parse status counts: {statuses}. spent ~${runner.spent:.2f}"
          + (" (list-price equivalent; a subscription is not billed per call)" if args.backend == "claude-code" else ""))
    if skipped:
        print(f"{skipped} calls not started because --max_cost {args.max_cost} was reached")
    if failed:
        print(f"{failed} calls failed and were not saved; rerun the same command to fill them in")
