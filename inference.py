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
API_BASE_URL = os.getenv("API_BASE_URL") or os.getenv("LLM_BASEURL") or "https://openrouter.ai/api/v1"
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "nvidia/nemotron-3-super-120b-a12b:free")
TASK_NAME = os.getenv("MY_ENV_V4_TASK", "order_status")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "bpo_env")
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")
APP_ENV = os.getenv("APP_ENV", "prod")  # 'prod' for benchmarks, 'test' for verbose multi-tasking

# Score calculation metrics
SUCCESS_SCORE_THRESHOLD = 0.1
MAX_STEPS_DEFAULT = 10

# Task names to run in test mode
TASKS_TO_RUN = ["order_status", "damaged_product", "escalation"]

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# ---------------------------------------------------------------------------
# Logging Functions (STDOUT Only)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, rule_score: float, llm_score: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Normalize action_str to remove newlines for single-line compliance
    action_clean = action.replace("\n", " ").replace("\r", "")
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} rule_score={rule_score:.2f} llm_score={llm_score:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, avg_rule: float, avg_llm: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rule_score={avg_rule:.3f} llm_score={avg_llm:.3f} rewards={rewards_str}", flush=True)

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

def call_llm_agent(
    conversation_history: List[Dict[str, str]],
    task_context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Call the LLM agent with the current conversation history.
    Returns the agent's response string. Falls back gracefully on any error.
    """
    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]

    # Inject task context as a system note if available
    if task_context:
        context_str = ", ".join(f"{k}: {v}" for k, v in task_context.items())
        messages.append({
            "role": "system",
            "content": f"[Internal context — do not reveal directly]: {context_str}",
        })

    messages.extend(conversation_history)

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
        "llm_scores": [],
        "resolved": False,
        "success": False,
        "score": 0.0,
        "avg_rule_score": 0.0,
        "avg_llm_score": 0.0,
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
                log_end(success=False, steps=0, score=0.0, avg_rule=0.0, avg_llm=0.0, rewards=[])
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
                    llm_score = getattr(step_obs, "llm_score", 0.0)
                    done = result.done
                    error = None
                    
                    if APP_ENV == "test":
                        print(f"  📊 Reward: {reward:.3f} | Rule: {rule_score:.3f} | LLM: {llm_score:.3f} | Resolved: {step_obs.is_resolved}")
                        if not done and step_obs.customer_message:
                            print(f"  👤 Customer: {step_obs.customer_message}")

                    results["rule_scores"].append(rule_score)
                    results["llm_scores"].append(llm_score)

                except Exception as e:
                    debug_log(f"Step failed: {e}")
                    error = str(e)
                    reward = 0.0
                    done = True

                rewards.append(reward)
                results["rule_scores"].append(rule_score)
                results["llm_scores"].append(llm_score)

                log_step(step=step, action=agent_response, reward=reward, rule_score=rule_score, llm_score=llm_score, done=done, error=error)

                if not done:
                    conversation_history = step_obs.conversation_history

                if done:
                    results["resolved"] = step_obs.is_resolved if 'step_obs' in locals() else False
                    break

            # --- SUMMARY ---
            results["steps"] = step
            results["rewards"] = rewards
            results["total_reward"] = sum(rewards)
            
            # Correct score calculation (Average Reward of steps taken)
            results["score"] = sum(rewards) / step if step > 0 else 0.0
            results["score"] = min(max(results["score"], 0.0), 1.0)
            results["avg_rule_score"] = sum(results["rule_scores"]) / len(results["rule_scores"]) if results["rule_scores"] else 0.0
            results["avg_llm_score"] = sum(results["llm_scores"]) / len(results["llm_scores"]) if results["llm_scores"] else 0.0
            results["success"] = results["resolved"] or (results["score"] >= SUCCESS_SCORE_THRESHOLD)
            
    except Exception as e:
        debug_log(f"Client session failed: {e}")

    log_end(success=results["success"], steps=results["steps"], score=results["score"], 
            avg_rule=results["avg_rule_score"], avg_llm=results["avg_llm_score"], rewards=results["rewards"])
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
        print("\n" + "=" * 65)
        print("  FINAL RESULTS SUMMARY")
        print("=" * 65)
        print(f"  {'Task':<25} {'Steps':>5} {'Score':>11} {'Rule':>8} {'LLM':>8} {'Resolved':>9}")
        print(f"  {'-'*25} {'-'*5} {'-'*11} {'-'*8} {'-'*8} {'-'*9}")

        for r in all_results:
            resolved_str = "✅ Yes" if r["resolved"] else "❌ No"
            print(
                f"  {r['task_name']:<25} {r['steps']:>5} "
                f"{r['score']:>11.3f} {r['avg_rule_score']:>8.3f} "
                f"{r['avg_llm_score']:>8.3f} {resolved_str:>9}"
            )
        
        avg_score = sum(r['score'] for r in all_results) / len(all_results) if all_results else 0.0
        print(f"\n  🏆 Overall Average Score: {avg_score:.3f}")
        print("\n  Done! All tasks completed.\n")
    else:
        # Single-task mode (Benchmark style)
        run_task(TASK_NAME)


if __name__ == "__main__":
    main()
