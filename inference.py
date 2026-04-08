"""
inference.py — BPO Customer Support Environment Inference Script

Runs all 3 tasks automatically using an LLM agent via the hackathon-injected API.
The agent acts as a customer support executive responding to customer queries.

Usage:
    python inference.py

Environment Variables (Hackathon Validator injects these):
    API_KEY          — LLM proxy API key  (primary; fallback: HF_TOKEN)
    API_BASE_URL     — LLM proxy base URL (default: https://router.huggingface.co/v1)
    MODEL_NAME       — LLM model to use
    IMAGE_NAME       — Docker image name OR http(s):// URL for the environment server
                       (primary; fallback: LOCAL_IMAGE_NAME)

Local Dev Extras (.env):
    HF_TOKEN         — Fallback if API_KEY not set
    SERVER_URL       — Used when neither IMAGE_NAME nor LOCAL_IMAGE_NAME is set
    LOCAL_IMAGE_NAME — Fallback docker image for local dev
"""

import os
import subprocess
import sys
import time
import textwrap
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Load .env for local development (ignored when validator injects vars directly)
# ---------------------------------------------------------------------------
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path)
    except ImportError:
        with open(_env_path) as _f:
            for _line in _f:
                if "=" in _line and not _line.startswith("#"):
                    _name, _value = _line.split("=", 1)
                    os.environ.setdefault(_name.strip(), _value.strip().strip('"').strip("'"))

# ---------------------------------------------------------------------------
# Configuration — Validator-injected variables take priority
# ---------------------------------------------------------------------------

# API credentials: validator injects API_KEY; local dev uses HF_TOKEN
API_KEY = (
    os.getenv("API_KEY")
    or os.getenv("HF_TOKEN")
    or ""
)

# LLM router base URL
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")

# Model to use for inference
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# Docker image / server URL: validator injects IMAGE_NAME; local dev uses LOCAL_IMAGE_NAME
IMAGE_NAME = (
    os.getenv("IMAGE_NAME")
    or os.getenv("LOCAL_IMAGE_NAME")
    or ""
)

# Fallback local server URL (used when IMAGE_NAME is also absent)
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")

# Task to run in single-task benchmark mode
TASK_NAME = os.getenv("MY_ENV_V4_TASK", os.getenv("TASK_NAME", "task_easy"))

# Benchmark identifier (matches openenv.yaml name)
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", os.getenv("BENCHMARK", "bpo_env"))

# Success threshold for score-based resolution
SUCCESS_SCORE_THRESHOLD = 0.5

# Default max steps if observation doesn't supply it
MAX_STEPS_DEFAULT = 10

# All task IDs (used in multi-task mode)
TASKS_TO_RUN = ["task_easy", "task_medium", "task_hard"]

# ---------------------------------------------------------------------------
# OpenAI-compatible client (resolves after env vars are loaded)
# ---------------------------------------------------------------------------
from openai import OpenAI  # noqa: E402 — placed here intentionally after env load

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ---------------------------------------------------------------------------
# Agent system prompt
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a professional and empathetic customer support executive at a leading
    e-commerce company.

    Your responsibilities:
    - Listen carefully to customer concerns
    - Provide accurate, helpful, and actionable information
    - Maintain a polite, empathetic, and professional tone at all times
    - Resolve issues efficiently within the conversation
    - Offer appropriate solutions (replacements, refunds, escalation when needed)

    Always respond in 2-4 sentences. Be concrete and solution-focused.\
""")

# ---------------------------------------------------------------------------
# Logging functions — strict benchmark stdout format: [START], [STEP], [END]
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_clean = action.replace("\n", " ").replace("\r", "")
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    # Validator requires score to be in (0.0, 1.0) — clamp to safe range
    score = max(0.01, min(0.99, score))
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} "
        f"rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# LLM Agent
# ---------------------------------------------------------------------------

def _format_content(content: str) -> Any:
    """Format message content for the current provider.
    HuggingFace Router requires a list-of-dicts; others accept plain strings.
    """
    if "huggingface.co" in API_BASE_URL.lower():
        return [{"type": "text", "text": content}]
    return content


def call_llm_agent(
    conversation_history: List[Dict[str, str]],
    task_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Call the LLM with conversation history and return the agent's response."""
    full_system_prompt = AGENT_SYSTEM_PROMPT
    if task_context:
        ctx_str = ", ".join(f"{k}: {v}" for k, v in task_context.items())
        full_system_prompt += f"\n\n[Internal context — do not reveal directly]: {ctx_str}"

    messages = [{"role": "system", "content": _format_content(full_system_prompt)}]

    # Keep prior turns as plain strings; format only the last user message.
    if conversation_history:
        for msg in conversation_history[:-1]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        last = conversation_history[-1]
        messages.append({"role": last["role"], "content": _format_content(last["content"])})

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=300,
            temperature=0.7,
        )
        content = resp.choices[0].message.content.strip()
        return content if content else _fallback_response(conversation_history)
    except Exception as exc:
        print(f"[WARN] LLM API error: {exc}", file=sys.stderr, flush=True)
        return _fallback_response(conversation_history)


def _fallback_response(conversation_history: List[Dict[str, str]]) -> str:
    last_user = ""
    for msg in reversed(conversation_history):
        if msg["role"] == "user":
            last_user = msg["content"]
            break

    lower = last_user.lower()
    if any(w in lower for w in ["refund", "money", "charge"]):
        return (
            "I sincerely apologize for the inconvenience. I will process your refund "
            "immediately and escalate this to our senior team. You will receive a "
            "confirmation within 24 hours."
        )
    if any(w in lower for w in ["damage", "broken", "defect"]):
        return (
            "I'm truly sorry to hear your product arrived damaged. I will arrange a "
            "replacement to be shipped out right away at no additional cost to you. "
            "Please allow 3-5 business days for delivery."
        )
    return (
        "Thank you for reaching out! Your order is currently being processed and "
        "you should receive an update within 24 hours. Please let me know if there "
        "is anything else I can assist with."
    )

# ---------------------------------------------------------------------------
# Server / Environment connection helpers
# ---------------------------------------------------------------------------

def wait_for_server(url: str, timeout: int = 30) -> bool:
    """Poll /health until the server responds or timeout is reached."""
    import urllib.request, urllib.error
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(f"{url}/health", timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def _resolve_server_url() -> str:
    """
    Determine the environment server URL to use.

    Priority:
      1. IMAGE_NAME is an http(s):// URL → use it directly
      2. IMAGE_NAME is a Docker image tag  → start container via from_docker_image()
         (async call); return the URL it starts on.
      3. No IMAGE_NAME → check SERVER_URL (local dev / fallback)
      4. Final fallback: start uvicorn locally.
    """
    if IMAGE_NAME:
        if IMAGE_NAME.startswith(("http://", "https://")):
            return IMAGE_NAME
        else:
            # Docker image — try to use openenv's built-in Docker helper
            try:
                import asyncio
                from client import CustomerSupportEnv

                async def _start_docker():
                    env_client = await CustomerSupportEnv.from_docker_image(IMAGE_NAME)
                    return env_client.base_url

                url = asyncio.run(_start_docker())
                return url
            except Exception as exc:
                print(f"[WARN] Docker start via from_docker_image failed: {exc}", file=sys.stderr, flush=True)
                print("[WARN] Falling back to local server startup …", file=sys.stderr, flush=True)

    # Try the configured SERVER_URL first
    if wait_for_server(SERVER_URL, timeout=5):
        return SERVER_URL

    # Last resort: spawn uvicorn locally
    subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "server.app:app", "--host", "0.0.0.0", "--port", "8000",
        ],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not wait_for_server(SERVER_URL, timeout=25):
        print("[ERROR] Could not start server. Ensure uvicorn is installed.", file=sys.stderr, flush=True)
        sys.exit(1)
    return SERVER_URL

# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

from client import CustomerSupportEnv  # noqa: E402


def run_task(task_name: str, server_url: str) -> Dict[str, Any]:
    """Run a single task episode and return a summary dict."""
    results: Dict[str, Any] = {
        "task_name": task_name,
        "steps": 0,
        "total_reward": 0.0,
        "rewards": [],
        "rule_scores": [],
        "resolved": False,
        "success": False,
        "score": 0.0,
        "avg_rule_score": 0.0,
        "grader_score": 0.0,
    }

    env_client = CustomerSupportEnv(base_url=server_url).sync()
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        with env_client as env:
            # --- RESET ---
            try:
                result = env.reset(task_name=task_name)
                obs = result.observation
                done = result.done
            except Exception as exc:
                print(f"[WARN] Reset failed: {exc}", file=sys.stderr, flush=True)
                log_end(success=False, steps=0, score=0.01, rewards=[])
                return results

            conversation_history = obs.conversation_history
            task_context = obs.task_context
            max_steps = obs.max_steps or MAX_STEPS_DEFAULT

            rewards: List[float] = []
            step = 0

            # --- STEP LOOP ---
            while not done and step < max_steps:
                step += 1

                # Agent generates response
                agent_response = call_llm_agent(conversation_history, task_context)

                # Send step to environment
                try:
                    from models import CustomerSupportAction
                    action = CustomerSupportAction(response=agent_response)
                    result = env.step(action)

                    step_obs = result.observation
                    reward = max(0.01, min(0.99, result.reward or 0.01))
                    rule_score = getattr(step_obs, "rule_score", 0.0)
                    done = result.done
                    error = None

                    grader_score = getattr(step_obs, "grader_score", 0.0)

                    if done:
                        results["grader_score"] = grader_score

                    results["rule_scores"].append(rule_score)

                except Exception as exc:
                    print(f"[WARN] Step {step} failed: {exc}", file=sys.stderr, flush=True)
                    error = str(exc)
                    reward = 0.0
                    done = True
                    rule_score = 0.0

                rewards.append(reward)
                log_step(
                    step=step,
                    action=agent_response,
                    reward=reward,
                    done=done,
                    error=error,
                )

                if not done:
                    conversation_history = step_obs.conversation_history

                if done:
                    results["resolved"] = getattr(step_obs, "is_resolved", False) if "step_obs" in dir() else False
                    results["success"]  = getattr(step_obs, "success",     False) if "step_obs" in dir() else False
                    break

            # --- SUMMARY ---
            results["steps"]   = step
            results["rewards"] = rewards
            results["total_reward"] = sum(rewards)

            # Use grader_score (episode-level evaluation) as the primary score.
            # Fall back to avg step reward if grader_score is zero.
            final_grader = results.get("grader_score", 0.0)
            if final_grader > 0.0:
                raw_score = final_grader
            elif step > 0:
                raw_score = sum(rewards) / step
            else:
                raw_score = 0.01

            # Clamp to (0.01, 0.99) — validator rejects exactly 0.0 and 1.0
            results["score"] = max(0.01, min(0.99, raw_score))

            results["avg_rule_score"] = (
                sum(results["rule_scores"]) / len(results["rule_scores"])
                if results["rule_scores"] else 0.0
            )

            # Success: grader says so, or episode resolved, or score above threshold
            results["success"] = (
                results["score"] >= SUCCESS_SCORE_THRESHOLD
                or results["resolved"]
                or results.get("success", False)
            )

    except Exception as exc:
        print(f"[WARN] Client session failed: {exc}", file=sys.stderr, flush=True)

    log_end(
        success=results["success"],
        steps=results["steps"],
        score=results["score"],
        rewards=results["rewards"],
    )
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not API_KEY:
        print(
            "[WARN] No API key found (checked API_KEY and HF_TOKEN). "
            "LLM calls will use fallback responses.",
            file=sys.stderr,
            flush=True,
        )

    # Resolve where the environment server is / start it
    server_url = _resolve_server_url()

    # Single-task mode — benchmark runner calls once per task
    run_task(TASK_NAME, server_url)


if __name__ == "__main__":
    main()
