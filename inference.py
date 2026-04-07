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
    APP_ENV          — 'prod' (default) for benchmark; 'test' for verbose multi-task
    DEBUG_MODE       — 'true' to print extra info to stderr in prod mode
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
TASK_NAME = os.getenv("MY_ENV_V4_TASK", os.getenv("TASK_NAME", "order_status"))

# Benchmark identifier (matches openenv.yaml name)
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", os.getenv("BENCHMARK", "bpo_env"))

# APP_ENV: 'prod' for strict benchmark output; 'test' for verbose multi-task output
APP_ENV = os.getenv("APP_ENV", "prod")

# DEBUG_MODE: emit extra info to stderr in prod mode
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

# Success threshold for score-based resolution
SUCCESS_SCORE_THRESHOLD = 0.5

# Default max steps if observation doesn't supply it
MAX_STEPS_DEFAULT = 10

# All task names (used in test/multi-task mode)
TASKS_TO_RUN = ["order_status", "damaged_product", "escalation"]

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
# Logging functions
# Benchmark stdout format: [START], [STEP], [END]
# Extra info goes to stderr (debug) or stdout only in test mode.
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    rule_score: float,
    done: bool,
    error: Optional[str],
    stage: str = "",
    mood: str = "",
    intent: str = "",
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_clean = action.replace("\n", " ").replace("\r", "")

    if APP_ENV == "test":
        extras = ""
        if stage:  extras += f" stage={stage}"
        if mood:   extras += f" mood={mood}"
        if intent: extras += f" intent={intent}"
        print(
            f"[STEP] step={step} action={action_clean} reward={reward:.2f} "
            f"rule_score={rule_score:.2f} done={done_val}"
            f"{extras} error={error_val}",
            flush=True,
        )
    else:
        # Prod/benchmark mode — mandatory fields only to stdout
        print(
            f"[STEP] step={step} action={action_clean} reward={reward:.2f} "
            f"done={done_val} error={error_val}",
            flush=True,
        )
        if DEBUG_MODE:
            extras = f"rule_score={rule_score:.2f}"
            if stage:  extras += f" stage={stage}"
            if mood:   extras += f" mood={mood}"
            if intent: extras += f" intent={intent}"
            print(f"   [DEBUG] STEP {step} EXTRAS: {extras}", file=sys.stderr, flush=True)


def log_end(
    success: bool,
    steps: int,
    score: float,
    avg_rule: float,
    rewards: List[float],
    grader_score: float = 0.0,
    failure_reason: str = "",
    reward_reason: str = "",
) -> None:
    # Validator requires score to be in (0.0, 1.0) — clamp to safe range
    score = max(0.01, min(0.99, score))
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    if APP_ENV == "test":
        failure_str = f" failure_reason={failure_reason}" if failure_reason and not success else ""
        reason_str  = f" reward_reason={reward_reason}" if reward_reason else ""
        print(
            f"[END] success={str(success).lower()} steps={steps} score={score:.3f} "
            f"rule_score={avg_rule:.3f} "
            f"grader_score={grader_score:.3f} rewards={rewards_str}{failure_str}{reason_str}",
            flush=True,
        )
    else:
        # Prod/benchmark mode — mandatory fields including score
        print(
            f"[END] success={str(success).lower()} steps={steps} score={score:.3f} "
            f"rewards={rewards_str}",
            flush=True,
        )
        if DEBUG_MODE:
            print(
                f"   [DEBUG] END SUMMARY: rule={avg_rule:.3f} grader={grader_score:.3f} "
                f"reason={failure_reason} reward_reason={reward_reason}",
                file=sys.stderr, flush=True,
            )


def debug_log(msg: str) -> None:
    if APP_ENV == "test":
        print(f"   {msg}", flush=True)
    elif DEBUG_MODE:
        print(f"   [DEBUG] {msg}", file=sys.stderr, flush=True)

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
        debug_log(f"Calling LLM: {MODEL_NAME} ...")
        t0 = time.time()
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=300,
            temperature=0.7,
        )
        debug_log(f"LLM response in {time.time()-t0:.2f}s")
        content = resp.choices[0].message.content.strip()
        return content if content else _fallback_response(conversation_history)
    except Exception as exc:
        debug_log(f"LLM API error: {exc}")
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
            debug_log(f"Environment URL from IMAGE_NAME: {IMAGE_NAME}")
            return IMAGE_NAME
        else:
            # Docker image — try to use openenv's built-in Docker helper
            debug_log(f"Starting Docker container from IMAGE_NAME={IMAGE_NAME} ...")
            try:
                import asyncio
                from client import CustomerSupportEnv

                async def _start_docker():
                    env_client = await CustomerSupportEnv.from_docker_image(IMAGE_NAME)
                    return env_client.base_url

                url = asyncio.run(_start_docker())
                debug_log(f"Docker environment started at: {url}")
                return url
            except Exception as exc:
                debug_log(f"Docker start via from_docker_image failed: {exc}")
                debug_log("Falling back to local server startup …")

    # Try the configured SERVER_URL first
    debug_log(f"Checking server at {SERVER_URL} ...")
    if wait_for_server(SERVER_URL, timeout=5):
        debug_log("Server already running.")
        return SERVER_URL

    # Last resort: spawn uvicorn locally
    debug_log("Starting local uvicorn server ...")
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
        debug_log("Could not start server. Ensure uvicorn is installed.")
        sys.exit(1)
    debug_log("Local server started.")
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
        "final_stage": "",
        "final_mood": "",
        "failure_reason": "",
        "reward_reason": "",
        "repetition_count": 0,
        "stall_count": 0,
    }

    if APP_ENV == "test":
        print(f"\n{'='*65}")
        print(f"  TASK: {task_name.upper().replace('_', ' ')}")
        print(f"{'='*65}")
    else:
        debug_log(f"Starting task: {task_name}")

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
                debug_log(f"Reset failed: {exc}")
                log_end(success=False, steps=0, score=0.01, avg_rule=0.0, rewards=[])
                return results

            if APP_ENV == "test":
                print(f"  👤 Customer: {obs.customer_message}\n")

            conversation_history = obs.conversation_history
            task_context = obs.task_context
            max_steps = obs.max_steps or MAX_STEPS_DEFAULT

            rewards: List[float] = []
            step = 0

            # --- STEP LOOP ---
            while not done and step < max_steps:
                step += 1
                if APP_ENV == "test":
                    print(f"  --- Step {step}/{max_steps} ---")

                # Agent generates response
                agent_response = call_llm_agent(conversation_history, task_context)
                if APP_ENV == "test":
                    print(f"  🤖 Agent: {agent_response}")

                # Send step to environment
                try:
                    from models import CustomerSupportAction
                    action = CustomerSupportAction(response=agent_response)
                    result = env.step(action)

                    step_obs = result.observation
                    reward = result.reward or 0.0
                    rule_score = getattr(step_obs, "rule_score", 0.0)
                    done = result.done
                    error = None

                    # State machine fields
                    stage = getattr(step_obs, "conversation_stage", "")
                    mood  = getattr(step_obs, "customer_mood", "")
                    intent = getattr(step_obs, "intent_detected", "")
                    grader_score = getattr(step_obs, "grader_score", 0.0)
                    reward_reason = getattr(step_obs, "reward_reason", "")
                    rep_count = getattr(step_obs, "repetition_count", 0)
                    stall_count = getattr(step_obs, "stall_count", 0)
                    failure_reason = getattr(step_obs, "failure_reason", "")

                    if done:
                        results["grader_score"]    = grader_score
                        results["reward_reason"]   = reward_reason
                        results["final_stage"]     = stage
                        results["final_mood"]      = mood
                        results["failure_reason"]  = failure_reason
                        results["repetition_count"] = rep_count
                        results["stall_count"]     = stall_count

                    if APP_ENV == "test":
                        rep_str   = f" ⚠️ REP={rep_count}"   if rep_count   > 0 else ""
                        stall_str = f" 🛑 STALL={stall_count}" if stall_count > 0 else ""
                        print(
                            f"  📊 Reward: {reward:.3f} | Rule: {rule_score:.3f} | "
                            f"Stage: {stage} | Mood: {mood} | Intent: {intent}"
                            f"{rep_str}{stall_str} | Resolved: {step_obs.is_resolved}"
                        )
                        if not done and step_obs.customer_message:
                            print(f"  👤 Customer: {step_obs.customer_message}")
                        if done:
                            is_success_obs = (
                                getattr(step_obs, "success", False)
                                or getattr(step_obs, "is_resolved", False)
                            )
                            success_str = "✅ SUCCESS" if is_success_obs else "❌ FAIL"
                            print(
                                f"  🏆 Grader: {grader_score:.3f} | {success_str}"
                                + (f" | Reason: {failure_reason}" if failure_reason else "")
                                + (f" | {reward_reason}" if reward_reason else "")
                            )

                    results["rule_scores"].append(rule_score)

                except Exception as exc:
                    debug_log(f"Step {step} failed: {exc}")
                    error = str(exc)
                    reward = 0.0
                    done = True
                    rule_score = 0.0
                    stage = mood = intent = ""

                rewards.append(reward)
                log_step(
                    step=step,
                    action=agent_response,
                    reward=reward,
                    rule_score=rule_score,
                    done=done,
                    error=error,
                    stage=stage,
                    mood=mood,
                    intent=intent,
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
        debug_log(f"Client session failed: {exc}")

    log_end(
        success=results["success"],
        steps=results["steps"],
        score=results["score"],
        avg_rule=results["avg_rule_score"],
        rewards=results["rewards"],
        grader_score=results.get("grader_score", 0.0),
        failure_reason=results.get("failure_reason", ""),
        reward_reason=results.get("reward_reason", ""),
    )
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    debug_log("=" * 60)
    debug_log("BPO Customer Support Environment — Inference Runner")
    debug_log(f"Model        : {MODEL_NAME}")
    debug_log(f"API_BASE_URL : {API_BASE_URL}")
    debug_log(f"IMAGE_NAME   : {IMAGE_NAME or '(not set)'}")
    debug_log(f"SERVER_URL   : {SERVER_URL}")
    debug_log(f"APP_ENV      : {APP_ENV}")
    debug_log("=" * 60)

    if not API_KEY:
        debug_log(
            "WARNING: No API key found (checked API_KEY and HF_TOKEN). "
            "LLM calls will use fallback responses."
        )

    # Resolve where the environment server is / start it
    server_url = _resolve_server_url()

    if APP_ENV == "test":
        # Multi-task mode — run all 3 tasks and print detailed report
        all_results = []
        for task_name in TASKS_TO_RUN:
            task_result = run_task(task_name, server_url)
            all_results.append(task_result)
            time.sleep(1)

        # Final report
        print("\n" + "=" * 100)
        print("  FINAL RESULTS SUMMARY")
        print("=" * 100)
        print(
            f"  {'Task':<25} {'Steps':>5} {'Score':>8} {'Grader':>7} {'Rule':>7} "
            f"{'Mood':>9} {'Success':>8} {'Failure Reason':<20}"
        )
        print(
            f"  {'-'*25} {'-'*5} {'-'*8} {'-'*7} {'-'*7} "
            f"{'-'*9} {'-'*8} {'-'*20}"
        )
        for r in all_results:
            success_str = "✅ Yes" if r["success"] else "❌ No"
            fail_str = r.get("failure_reason", "") or "—"
            print(
                f"  {r['task_name']:<25} {r['steps']:>5} "
                f"{r['score']:>8.3f} {r.get('grader_score', 0.0):>7.3f} "
                f"{r['avg_rule_score']:>7.3f} "
                f"{r.get('final_mood', 'n/a'):>9} {success_str:>8} {fail_str:<20}"
            )

        avg_score  = sum(r["score"] for r in all_results) / len(all_results) if all_results else 0.0
        avg_grader = sum(r.get("grader_score", 0.0) for r in all_results) / len(all_results) if all_results else 0.0
        success_count = sum(1 for r in all_results if r["success"])
        print(f"\n  🏆 Overall Average Score:  {avg_score:.3f}")
        print(f"  🎯 Overall Grader Score:   {avg_grader:.3f}")
        print(f"  ✅ Tasks Succeeded:        {success_count}/{len(all_results)}")
        print("\n  Done! All tasks completed.\n")
    else:
        # Single-task mode — benchmark runner calls once per task
        run_task(TASK_NAME, server_url)


if __name__ == "__main__":
    main()
