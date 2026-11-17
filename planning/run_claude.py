"""
Send planning questions from subsets/ to Claude and save the raw answers.

Default backend is headless Claude Code (`claude -p`) on the logged-in Claude
subscription, set up like in ../theory_of_mind/tombench/run_claude.py: empty
working directory, all tools off, advisor tool off, benchmark prompt as the
system prompt. `--backend api` uses the Messages API instead (needs a key).

Usage:
    python run_claude.py --questions subsets/blocksworld.jsonl --limit 20
    python run_claude.py --questions subsets/natural_plan_trip.jsonl --limit 24 --workers 4

Results go to results/<questions file name>.jsonl by default. Re-running the same
command skips questions that are already answered, so it resumes after a usage limit.
Score the answers with score.py.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

# Settings that must match for rows to be appended to an existing results file.
RUN_KEYS = ["backend", "model", "effort", "questions_file"]

# Removed from the environment of `claude -p`, so it runs on the logged-in
# subscription and not on API credits, and does not detect a parent session.
CLAUDE_CODE_DROP_ENV = {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT"}

DEFAULT_KEY_FILE = os.path.expanduser("~/.config/tombench/anthropic_api_key")


class CallFailed(Exception):
    pass


class Runner:
    def __init__(self, args, settings):
        self.args = args
        self.settings = settings
        self.lock = threading.Lock()
        if args.backend == "api":
            import anthropic

            key = Path(args.api_key_file).read_text().strip() if os.path.exists(args.api_key_file) else None
            self.client = anthropic.Anthropic(api_key=key, max_retries=args.max_retries)
        else:
            self.claude_bin = args.claude_bin or shutil.which("claude")
            if not self.claude_bin:
                sys.exit("Could not find 'claude' on PATH. Install Claude Code and log in, or pass --claude_bin.")
            self.workdir = tempfile.mkdtemp(prefix="planning_claude_code_")  # empty folder, nothing to peek at
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
        return {
            "text": "".join(b.text for b in response.content if b.type == "text"),
            "stop_reason": response.stop_reason,
            "usage": {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens},
            "cost_usd": None,
            "response_model": response.model,
        }

    def call_claude_code(self, system, prompt):
        # The prompt goes through stdin: 5-shot Natural Plan prompts are too long
        # for a Windows command line.
        cmd = [self.claude_bin, "-p", "--output-format", "json",
               "--model", self.args.model, "--effort", self.args.effort,
               "--system-prompt", system, "--tools", "", "--safe-mode",
               "--strict-mcp-config", "--no-session-persistence"]
        try:
            proc = subprocess.run(cmd, input=prompt, cwd=self.workdir, env=self.env, capture_output=True,
                                  text=True, encoding="utf-8", errors="replace", timeout=self.args.timeout)
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
            "usage": {
                "input_tokens": usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0)
                + usage.get("cache_read_input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "thinking_tokens": (usage.get("output_tokens_details") or {}).get("thinking_tokens"),
            },
            # List-price equivalent reported by Claude Code; not billed on a subscription.
            "cost_usd": data.get("total_cost_usd"),
            "response_model": models[0] if len(models) == 1 else models,
            "duration_s": round((data.get("duration_ms") or 0) / 1000, 1),
            # Evidence that no tools ran: one turn, no web requests, no blocked tool calls.
            "num_turns": data.get("num_turns"),
            "web_requests": sum((usage.get("server_tool_use") or {}).values()),
            "permission_denials": len(data.get("permission_denials") or []),
        }

    def ask(self, q):
        if self.args.backend == "api":
            result = self.call_api(q["system"], q["prompt"])
        else:
            result = self.call_claude_code(q["system"], q["prompt"])
        row = dict(self.settings)
        row.update({"id": q["id"], "benchmark": q["benchmark"], **result,
                    "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
        with self.lock:
            with open(self.args.output, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row


def load_done(path, settings):
    done = set()
    if not os.path.exists(path):
        return done
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for k in RUN_KEYS:
            if row.get(k) != settings[k]:
                sys.exit(f"{path} holds results with {k}={row.get(k)!r}, this run has {settings[k]!r}; use a different --output")
        done.add(row["id"])
    return done


def main():
    parser = argparse.ArgumentParser(description="Run planning questions against a Claude model.")
    parser.add_argument("--questions", required=True, help="a file in subsets/")
    parser.add_argument("--output", default="", help="default: results/<questions file name>")
    parser.add_argument("--backend", default="claude-code", choices=["claude-code", "api"])
    parser.add_argument("--model", default="claude-opus-5-5")
    parser.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--limit", type=int, default=0, help="only the first N questions (they are a balanced sample)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1200, help="claude-code backend: seconds per call")
    parser.add_argument("--claude_bin", default="", help="path to claude(.exe); default: the one on PATH")
    parser.add_argument("--max_tokens", type=int, default=32000, help="api backend only")
    parser.add_argument("--max_retries", type=int, default=8, help="api backend only")
    parser.add_argument("--api_key_file", default=DEFAULT_KEY_FILE, help="api backend only; falls back to ANTHROPIC_API_KEY")
    parser.add_argument("--dry_run", action="store_true", help="print the first prompt without calling the model")
    args = parser.parse_args()

    questions = [json.loads(l) for l in Path(args.questions).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        questions = questions[: args.limit]
    if args.dry_run:
        q = questions[0]
        print(f"=== {q['id']} ===\n[system]\n{q['system']}\n[user]\n{q['prompt']}")
        return

    args.output = args.output or str(Path(__file__).parent / "results" / Path(args.questions).name)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    settings = {"backend": args.backend, "model": args.model, "effort": args.effort,
                "questions_file": Path(args.questions).name}
    done = load_done(args.output, settings)
    todo = [q for q in questions if q["id"] not in done]
    print(f"{len(questions)} questions: {len(done & {q['id'] for q in questions})} already done, {len(todo)} to run ({args.backend})")

    runner = Runner(args, settings)
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(runner.ask, q): q["id"] for q in todo}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            try:
                fut.result()
            except Exception as e:
                failed += 1
                tqdm.write(f"{futures[fut]} failed: {type(e).__name__}: {e}")
                if "Not logged in" in str(e) or "usage limit" in str(e).lower():
                    pool.shutdown(wait=False, cancel_futures=True)
                    sys.exit("stopping: " + str(e))
    print(f"done. results in {args.output}")
    if failed:
        print(f"{failed} calls failed and were not saved; rerun the same command to fill them in")


if __name__ == "__main__":
    main()
