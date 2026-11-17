"""Claude Code CLI adapter: runs a model through your Claude subscription login.

Calls `claude -p` (headless Claude Code) once per attempt, with:
  * no API key: ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN are removed from the
    child environment, so the CLI uses your `claude` login (Pro/Max/Team)
  * no tools: `--tools ""` disables every built-in tool (web search, web
    fetch, bash, file tools, subagents), plus no MCP servers and no Chrome
  * no customisation: `--safe-mode` skips CLAUDE.md, skills, plugins, hooks,
    and a short neutral system prompt replaces Claude Code's agent prompt
  * an empty, throwaway working directory per attempt

The model only reasons (with extended thinking at the configured effort) and
answers in text, which makes it close to a plain single API call.
"""
from .provider import ProviderAdapter
from arc_agi_benchmarking.schemas import (
    AttemptMetadata,
    Choice,
    Message,
    Usage,
    Cost,
    CompletionTokensDetails,
    PromptTokensDetails,
    Attempt,
)
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import subprocess
import tempfile
import logging
import shutil
import json
import os
import re

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# Credentials that would make the CLI bill an API account instead of the login.
_STRIPPED_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
)


class ClaudecodecliAdapter(ProviderAdapter):
    request_limit_scope = "adapter_invocation"

    def init_client(self):
        """No API key is needed; just make sure the `claude` CLI can be found."""
        self._executable = self._resolve_executable()
        return None

    # ------------------------------------------------------------------ run
    def make_prediction(
        self,
        prompt: str,
        task_id: Optional[str] = None,
        test_id: Optional[str] = None,
        pair_index: int = None,
    ) -> Attempt:
        start_time = datetime.now(timezone.utc)
        run = self._request(
            "claude_code.print_session",
            self._run_claude,
            prompt,
            _raw_log_request={
                "prompt": prompt,
                "model": self.model_config.model_name,
                "effort": self.model_config.kwargs.get("effort"),
                "task_id": task_id,
                "test_id": test_id,
                "pair_index": pair_index,
            },
        )
        end_time = datetime.now(timezone.utc)

        usage = run.get("usage") or {}
        cache_creation = usage.get("cache_creation_input_tokens", 0) or 0
        cache_read = usage.get("cache_read_input_tokens", 0) or 0
        prompt_tokens = (usage.get("input_tokens", 0) or 0) + cache_creation + cache_read
        completion_tokens = usage.get("output_tokens", 0) or 0
        thinking_tokens = (usage.get("output_tokens_details") or {}).get("thinking_tokens", 0) or 0

        # Notional cost at API list prices from models.yml (your subscription
        # is not charged per token; this is just for comparison with API runs).
        prompt_cost = prompt_tokens * self.model_config.pricing.input / 1_000_000
        completion_cost = completion_tokens * self.model_config.pricing.output / 1_000_000

        answer = run.get("result") or ""
        metadata = AttemptMetadata(
            model=self.model_config.model_name,
            provider=self.model_config.provider,
            start_timestamp=start_time,
            end_timestamp=end_time,
            choices=[
                Choice(index=0, message=Message(role="user", content=prompt)),
                Choice(index=1, message=Message(role="assistant", content=answer)),
            ],
            kwargs=self.model_config.kwargs,
            reasoning_summary=None,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                prompt_tokens_details=PromptTokensDetails(
                    cache_creation_tokens=cache_creation,
                    cache_read_tokens=cache_read,
                ),
                completion_tokens_details=CompletionTokensDetails(
                    reasoning_tokens=thinking_tokens,
                    accepted_prediction_tokens=completion_tokens,
                    rejected_prediction_tokens=0,
                ),
            ),
            cost=Cost(
                prompt_cost=prompt_cost,
                completion_cost=completion_cost,
                total_cost=prompt_cost + completion_cost,
            ),
            task_id=task_id,
            pair_index=pair_index,
            test_id=test_id,
            tool_calls=None,
            num_turns=run.get("num_turns") or 1,
        )
        return Attempt(metadata=metadata, answer=answer)

    def _build_args(self) -> List[str]:
        kw = self.model_config.kwargs
        args = [
            self._executable,
            "-p",
            "--output-format", "json",
            "--model", self.model_config.model_name,
            "--tools", "",                 # no built-in tools at all (incl. WebSearch/WebFetch)
            "--strict-mcp-config",         # no MCP servers
            "--safe-mode",                 # no CLAUDE.md, skills, plugins, hooks
            "--disable-slash-commands",
            "--no-chrome",
            "--no-session-persistence",
            "--permission-mode", "dontAsk",
            "--system-prompt", kw.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
        ]
        if kw.get("effort"):
            args += ["--effort", str(kw["effort"])]
        return args

    def _run_claude(self, prompt: str) -> Dict[str, Any]:
        env = {k: v for k, v in os.environ.items() if k not in _STRIPPED_ENV}
        timeout = self.model_config.kwargs.get("cli_timeout_seconds", 3600)

        with tempfile.TemporaryDirectory(prefix="arc_claude_") as workdir:
            proc = subprocess.run(
                self._build_args(),
                input=prompt,
                capture_output=True,
                text=True,
                env=env,
                cwd=workdir,
                timeout=timeout,
            )

        out = proc.stdout.strip()
        try:
            data = json.loads(out.splitlines()[-1]) if out else None
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Could not parse Claude Code output: {out[:500]}") from exc
        if data is None:
            raise RuntimeError(
                f"Claude Code exited with code {proc.returncode} and no output: {proc.stderr.strip()[:500]}"
            )
        if data.get("is_error"):
            raise RuntimeError(f"Claude Code error: {data.get('result')}")

        # Sanity checks: the run must have been a single, tool-free turn.
        server_tools = (data.get("usage") or {}).get("server_tool_use") or {}
        if any(server_tools.get(k) for k in ("web_search_requests", "web_fetch_requests")):
            raise RuntimeError(f"Web tools were used despite being disabled: {server_tools}")
        if (data.get("num_turns") or 1) > 1:
            logger.warning(f"Expected 1 turn, got {data.get('num_turns')}")
        models_used = list((data.get("modelUsage") or {}).keys())
        if models_used:
            logger.info(f"Claude Code models used: {models_used}")
        return data

    def _resolve_executable(self) -> str:
        override = self.model_config.kwargs.get("claude_cli_path") or os.environ.get("CLAUDE_CLI_PATH")
        if override:
            return override
        exe = shutil.which("claude")
        if not exe:
            raise FileNotFoundError(
                "claude executable not found in PATH. Install Claude Code and run `claude` once to log in "
                "(or set CLAUDE_CLI_PATH)."
            )
        return exe

    # --------------------------------------------------------------- parsing
    def chat_completion(self, messages, tools=[]):
        raise NotImplementedError("Claude Code CLI adapter uses `claude -p`, not chat_completion")

    def extract_json_from_response(self, input_response: str) -> List[List[int]]:
        """Return the last well-formed integer grid in the answer (models usually end with it)."""
        candidates = re.findall(r'\[\s*\[[\d\s,\[\]]+\]\s*\]', input_response)
        for match in reversed(candidates):
            try:
                parsed = json.loads(match)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, list) and parsed and all(
                isinstance(row, list) and all(isinstance(x, int) for x in row) for row in parsed
            ):
                return parsed
        return None
