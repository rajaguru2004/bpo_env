"""
inference.py — BPO Customer Support Environment Inference Script

Runs all 3 tasks automatically using an LLM agent via OpenRouter API.
The agent acts as a customer support executive responding to customer queries.

Usage:
    python inference.py

Environment Variables Required:
    OPENAI_API_KEY   — Your OpenRouter API key
    LLM_BASEURL      — https://openrouter.ai/api/v1
    MODEL_NAME       — e.g. nvidia/nemotron-3-super-120b-a12b:free
    SERVER_URL       — (optional) defaults to http://localhost:8000
"""

import os
import subprocess
import sys
import time
import textwrap
from typing import Any, Dict, List, Optional

# Load environment variables from .env if it exists
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_path):
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        with open(env_path) as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    name, value = line.split("=", 1)
                    os.environ[name.strip()] = value.strip().strip('"').strip("'")

# ---------------------------------------------------------------------------
# Constants & Mandatory Benchmark Variables
# ---------------------------------------------------------------------------
from openai import OpenAI

# Required by the Benchmark runner
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3.5-35B-A3B:novita")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

TASK_NAME = os.getenv("MY_ENV_V4_TASK", os.getenv("TASK_NAME", "order_status"))
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", os.getenv("BENCHMARK", "bpo_env"))
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")
APP_ENV = os.getenv("APP_ENV", "prod")  # 'prod' for benchmarks, 'test' for verbose multi-tasking

# Score calculation metrics — success requires genuine resolution, not just threshold
MAX_STEPS_DEFAULT = 10
SUCCESS_SCORE_THRESHOLD = 0.5  # normalized score in [0, 1]

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# ---------------------------------------------------------------------------
# Logging Functions (STDOUT for Benchmark, STDERR for Debug)
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
    # Clean action string for logs (no raw newlines)
    action_clean = action.replace("\n", " ").replace("\r", "")
    
    if APP_ENV == "test":
        # VERBOSE LOGGING FOR TEST MODE
        extras = ""
        if stage: extras += f" stage={stage}"
        if mood: extras += f" mood={mood}"
        if intent: extras += f" intent={intent}"
        print(
            f"[STEP] step={step} action={action_clean} reward={reward:.2f} "
            f"rule_score={rule_score:.2f} done={done_val}"
            f"{extras} error={error_val}",
            flush=True,
        )
    else:
        # OUTPUT MANDATORY FIELDS ONLY TO STDOUT (Benchmark Mode)
        print(
            f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
            flush=True,
        )
        # EXTRA STATE INFO TO STDERR
        extras = f"rule_score={rule_score:.2f}"
        if stage: extras += f" stage={stage}"
        if mood: extras += f" mood={mood}"
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
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    
    if APP_ENV == "test":
        # VERBOSE SUMMARY FOR TEST MODE
        failure_str = f" failure_reason={failure_reason}" if failure_reason and not success else ""
        reason_str  = f" reward_reason={reward_reason}" if reward_reason else ""
        print(
            f"[END] success={str(success).lower()} steps={steps} score={score:.3f} "
            f"rule_score={avg_rule:.3f} "
            f"grader_score={grader_score:.3f} rewards={rewards_str}{failure_str}{reason_str}",
            flush=True,
        )
    else:
        # OUTPUT MANDATORY FIELDS ONLY TO STDOUT (Benchmark Mode)
        print(
            f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
            flush=True,
        )
        # EXTRA SUMMARY INFO TO STDERR
        print(
            f"   [DEBUG] END SUMMARY: rule={avg_rule:.3f} grader={grader_score:.3f} reason={failure_reason} reward_reason={reward_reason}",
            file=sys.stderr, flush=True
        )

# Helper for descriptive logs
def debug_log(msg: str) -> None:
    # If in test mode, print to stdout (old way). If prod, stderr only.
    if APP_ENV == "test":
        print(f"   {msg}", flush=True)
    else:
        print(f"   [DEBUG] {msg}", file=sys.stderr, flush=True)

# Task names to run
TASKS_TO_RUN = ["order_status", "damaged_product", "escalation"]

AGENT_SYSTEM_PROMPT = """You are a professional and empathetic customer support executive at a leading e-commerce company.

Your responsibilities:
- Listen carefully to customer concerns
- Provide accurate, helpful, and actionable information
- Maintain a polite, empathetic, and professional tone at all times
- Resolve issues efficiently within the conversation
- Offer appropriate solutions (replacements, refunds, escalation when needed)

Always respond in 2-4 sentences. Be concrete and solution-focused."""


# ---------------------------------------------------------------------------
# LLM Agent Call
# ---------------------------------------------------------------------------

def _format_content(content: str) -> Any:
    """
    Format message content for the current provider.
    HuggingFace Router requires content as a list of dicts.
    Others (OpenRouter, Local) usually accept both.
    """
    is_hf = "huggingface.co" in API_BASE_URL.lower()
    if is_hf:
        return [{"type": "text", "text": content}]
    return content


def call_llm_agent(
    conversation_history: List[Dict[str, str]],
    task_context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Call the LLM agent with the current conversation history.
    Returns the agent's response string. Falls back gracefully on any error.
    """
    # Consolidate system instruction with task context for reliability
    full_system_prompt = AGENT_SYSTEM_PROMPT
    if task_context:
        context_str = ", ".join(f"{k}: {v}" for k, v in task_context.items())
        full_system_prompt += f"\n\n[Internal context — do not reveal directly]: {context_str}"

    messages = [{"role": "system", "content": _format_content(full_system_prompt)}]

    # For conversation history: keep all prior turns as plain strings for proper
    # multi-turn context. Only format the LAST user message for HF Router compatibility.
    # Together AI's backend loses context when the entire history uses list-of-dicts.
    if conversation_history:
        # All messages except the last get plain string content
        for msg in conversation_history[:-1]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"],  # Always plain string for history
            })
        # Last message (current customer input) gets provider-specific format
        last_msg = conversation_history[-1]
        messages.append({
            "role": last_msg["role"],
            "content": _format_content(last_msg["content"]),
        })

    try:
        debug_log(f"Calling LLM Agent: {MODEL_NAME} ...")
        start_time = time.time()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=300,
            temperature=0.7,
        )
        duration = time.time() - start_time
        debug_log(f"LLM Response Received: {duration:.2f}s")

        content = response.choices[0].message.content.strip()
        return content if content else _fallback_response(conversation_history)

    except Exception as e:
        debug_log(f"API Error: {e}")
        return _fallback_response(conversation_history)


def _fallback_response(conversation_history: List[Dict[str, str]]) -> str:
    """Polite fallback response when API call fails."""
    last_user_msg = ""
    for msg in reversed(conversation_history):
        if msg["role"] == "user":
            last_user_msg = msg["content"]
            break

    if any(word in last_user_msg.lower() for word in ["refund", "money", "charge"]):
        return (
            "I sincerely apologize for the inconvenience. I will process your refund "
            "request immediately and escalate this to our senior team. You will receive "
            "a confirmation within 24 hours."
        )
    elif any(word in last_user_msg.lower() for word in ["damage", "broken", "defect"]):
        return (
            "I'm truly sorry to hear your product arrived damaged. I will arrange a "
            "replacement to be shipped out right away at no additional cost to you. "
            "Please allow 3-5 business days for delivery."
        )
    else:
        return (
            "Thank you for reaching out! I'd be happy to help you with your concern. "
            "Your order is currently being processed and you should receive an update "
            "within 24 hours. Please let me know if there's anything else I can assist with."
        )


# ---------------------------------------------------------------------------
# Server Health Check
# ---------------------------------------------------------------------------

def wait_for_server(url: str, timeout: int = 30) -> bool:
    """Wait for the server to become available."""
    import urllib.request
    import urllib.error

    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(f"{url}/health", timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


# ---------------------------------------------------------------------------
# Task Runner
# ---------------------------------------------------------------------------

from client import CustomerSupportEnv

def run_task(task_name: str) -> Dict[str, Any]:
    """
    Run a single task episode using the CustomerSupportEnv WebSocket client.
    Returns a summary dict with scores and steps.
    """
    results = {
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

    # Use the WebSocket client with sync wrapper
    client_env = CustomerSupportEnv(base_url=SERVER_URL).sync()
    
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        with client_env as env:
            # --- RESET ---
            try:
                result = env.reset(task_name=task_name)
                obs = result.observation
                done = result.done
            except Exception as e:
                debug_log(f"Reset failed: {e}")
                log_end(success=False, steps=0, score=0.0, avg_rule=0.0, rewards=[])
                return results

            if APP_ENV == "test":
                print(f"  👤 Customer: {obs.customer_message}\n")

            conversation_history = obs.conversation_history
            task_context = obs.task_context
            max_steps = obs.max_steps or MAX_STEPS_DEFAULT
            
            rewards = []
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
                    mood = getattr(step_obs, "customer_mood", "")
                    intent = getattr(step_obs, "intent_detected", "")
                    grader_score = getattr(step_obs, "grader_score", 0.0)
                    reward_reason = getattr(step_obs, "reward_reason", "")
                    rep_count = getattr(step_obs, "repetition_count", 0)
                    stall_count = getattr(step_obs, "stall_count", 0)
                    failure_reason = getattr(step_obs, "failure_reason", "")

                    if done:
                        results["grader_score"] = grader_score
                        results["reward_reason"] = reward_reason
                        results["final_stage"] = stage
                        results["final_mood"] = mood
                        results["failure_reason"] = failure_reason
                        results["repetition_count"] = rep_count
                        results["stall_count"] = stall_count

                    if APP_ENV == "test":
                        rep_str = f" ⚠️ REP={rep_count}" if rep_count > 0 else ""
                        stall_str = f" 🛑 STALL={stall_count}" if stall_count > 0 else ""
                        print(
                            f"  📊 Reward: {reward:.3f} | Rule: {rule_score:.3f} | "
                            f"Stage: {stage} | Mood: {mood} | Intent: {intent}"
                            f"{rep_str}{stall_str} | Resolved: {step_obs.is_resolved}"
                        )
                        if not done and step_obs.customer_message:
                            print(f"  👤 Customer: {step_obs.customer_message}")
                        if done:
                            # success field may not propagate via client — fall back to is_resolved
                            is_success = getattr(step_obs, 'success', False) or getattr(step_obs, 'is_resolved', False)
                            success_str = "✅ SUCCESS" if is_success else "❌ FAIL"
                            print(f"  🏆 Grader: {grader_score:.3f} | {success_str}"
                                  + (f" | Reason: {failure_reason}" if failure_reason else "")
                                  + (f" | {reward_reason}" if reward_reason else ""))

                    results["rule_scores"].append(rule_score)

                except Exception as e:
                    debug_log(f"Step failed: {e}")
                    error = str(e)
                    reward = 0.0
                    done = True
                    rule_score = 0.0
                    stage = ""
                    mood = ""
                    intent = ""

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
                    results["resolved"] = step_obs.is_resolved if 'step_obs' in locals() else False
                    results["success"] = getattr(step_obs, 'success', False) if 'step_obs' in locals() else False
                    break

            # --- SUMMARY ---
            results["steps"] = step
            results["rewards"] = rewards
            results["total_reward"] = sum(rewards)
            
            # Score = average per-step reward (independent of grader_score).
            # grader_score is the separate episode-level evaluation from grade_episode().
            raw_score = sum(rewards) / step if step > 0 else 0.0
            results["score"] = min(1.0, max(0.0, raw_score))
                
            results["avg_rule_score"] = sum(results["rule_scores"]) / len(results["rule_scores"]) if results["rule_scores"] else 0.0
            
            # Success is determined by score threshold OR manual resolution flag
            results["success"] = results["score"] >= SUCCESS_SCORE_THRESHOLD or results["resolved"]
            
    except Exception as e:
        debug_log(f"Client session failed: {e}")

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
    # Show configuration in stderr/stdout
    debug_log("=" * 60)
    debug_log("BPO Customer Support Environment — Inference Runner")
    debug_log(f"Model   : {MODEL_NAME}")
    debug_log(f"Server  : {SERVER_URL}")
    debug_log(f"Env     : {APP_ENV}")
    debug_log("=" * 60)

    # Validate env vars
    if not HF_TOKEN:
        debug_log("HF_TOKEN (API Key) is not set. Please export your API key.")
        sys.exit(1)

    # Check server health
    debug_log(f"Checking server at {SERVER_URL} ...")
    if not wait_for_server(SERVER_URL, timeout=10):
        debug_log(f"Server at {SERVER_URL} not responding. Attempting to start it...")
        # Try to start server as a subprocess
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if not wait_for_server(SERVER_URL, timeout=20):
            debug_log("Could not start server. Please run: uvicorn server.app:app --port 8000")
            sys.exit(1)
        debug_log("Server started successfully.")
    else:
        debug_log("Server is up!")

    if APP_ENV == "test":
        # Multi-task mode (The Old Way)
        all_results = []
        for task_name in TASKS_TO_RUN:
            task_result = run_task(task_name)
            all_results.append(task_result)
            time.sleep(1)

        # Final Report
        print("\n" + "=" * 100)
        print("  FINAL RESULTS SUMMARY")
        print("=" * 100)
        print(f"  {'Task':<25} {'Steps':>5} {'Score':>8} {'Grader':>7} {'Rule':>7} "
              f"{'Mood':>9} {'Success':>8} {'Failure Reason':<20}")
        print(f"  {'-'*25} {'-'*5} {'-'*8} {'-'*7} {'-'*7} "
              f"{'-'*9} {'-'*8} {'-'*20}")

        for r in all_results:
            success_str = "✅ Yes" if r["success"] else "❌ No"
            fail_str = r.get("failure_reason", "") or "—"
            print(
                f"  {r['task_name']:<25} {r['steps']:>5} "
                f"{r['score']:>8.3f} {r.get('grader_score', 0.0):>7.3f} "
                f"{r['avg_rule_score']:>7.3f} "
                f"{r.get('final_mood', 'n/a'):>9} {success_str:>8} {fail_str:<20}"
            )

        avg_score = sum(r["score"] for r in all_results) / len(all_results) if all_results else 0.0
        avg_grader = sum(r.get("grader_score", 0.0) for r in all_results) / len(all_results) if all_results else 0.0
        success_count = sum(1 for r in all_results if r["success"])
        print(f"\n  🏆 Overall Average Score:  {avg_score:.3f}")
        print(f"  🎯 Overall Grader Score:   {avg_grader:.3f}")
        print(f"  ✅ Tasks Succeeded:        {success_count}/{len(all_results)}")
        print("\n  Done! All tasks completed.\n")
    else:
        # Single-task mode (Benchmark style)
        run_task(TASK_NAME)


if __name__ == "__main__":
    main()
