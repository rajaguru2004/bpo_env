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
from typing import Any, Dict, List, Optional

# Load environment variables from .env if it exists
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_path):
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        # Simple manual fallback for .env loading
        with open(env_path) as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    name, value = line.split("=", 1)
                    os.environ[name.strip()] = value.strip().strip('"').strip("'")

# ---------------------------------------------------------------------------
# LLM Client Setup
# ---------------------------------------------------------------------------

from openai import OpenAI

LLM_BASEURL = os.getenv("LLM_BASEURL", "https://openrouter.ai/api/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "nvidia/nemotron-3-super-120b-a12b:free")
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")

client = OpenAI(
    base_url=LLM_BASEURL,
    api_key=OPENAI_API_KEY,
)

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
        print(f"   [⏳ Calling LLM Agent]: {MODEL_NAME} ...")
        start_time = time.time()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=300,
            temperature=0.7,
            extra_body={"reasoning": {"enabled": True}},
        )
        duration = time.time() - start_time
        print(f"   [✅ LLM Response Received]: {duration:.2f}s")

        # Preserve reasoning details if present
        choice = response.choices[0]
        reasoning = getattr(choice.message, "reasoning_content", None)
        if reasoning:
            print(f"   [🧠 Reasoning]: {reasoning[:120]}...")

        content = choice.message.content.strip()
        return content if content else _fallback_response(conversation_history)

    except Exception as e:
        print(f"   [⚠️  API Error]: {e}")
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
        "avg_reward": 0.0,
        "avg_rule_score": 0.0,
        "avg_llm_score": 0.0,
        "resolved": False,
        "step_logs": [],
    }

    print(f"\n{'='*65}")
    print(f"  TASK: {task_name.upper().replace('_', ' ')}")
    print(f"{'='*65}")

    # Use the WebSocket client with sync wrapper
    client_env = CustomerSupportEnv(base_url=SERVER_URL).sync()
    
    try:
        with client_env as env:
            # --- RESET ---
            try:
                result = env.reset(task_name=task_name)
                obs = result.observation
            except Exception as e:
                print(f"  [ERROR] Reset failed: {e}")
                return results

            customer_message = obs.customer_message
            task_context = obs.task_context
            max_steps = obs.max_steps
            conversation_history = obs.conversation_history

            print(f"\n  📋 Task Context: {task_context}")
            print(f"  👤 Customer: {customer_message}\n")

            rule_scores = []
            llm_scores = []
            done = False
            step = 0

            # --- STEP LOOP ---
            while not done and step < max_steps:
                step += 1
                print(f"  --- Step {step}/{max_steps} ---")

                # Agent generates response
                agent_response = call_llm_agent(conversation_history, task_context)
                print(f"  🤖 Agent: {agent_response}")

                # Send step to environment
                try:
                    from models import CustomerSupportAction
                    action = CustomerSupportAction(response=agent_response)
                    
                    print(f"   [📡 Sending action to server] ...")
                    result = env.step(action)
                    print(f"   [📥 Received response]: reward={result.reward:.3f}, done={result.done}")
                    
                    step_obs = result.observation
                    reward = result.reward or 0.0
                    done = result.done
                except Exception as e:
                    print(f"  [ERROR] Step failed: {e}")
                    break

                rule_score = getattr(step_obs, "rule_score", 0.0)
                llm_score = getattr(step_obs, "llm_score", 0.0)
                next_customer_msg = step_obs.customer_message
                is_resolved = step_obs.is_resolved

                rule_scores.append(rule_score)
                llm_scores.append(llm_score)

                print(f"  📊 Reward: {reward:.3f} | Rule: {rule_score:.3f} | LLM: {llm_score:.3f} | Resolved: {is_resolved}")

                if not done and next_customer_msg:
                    print(f"  👤 Customer: {next_customer_msg}")
                    # Update conversation history for LLM context
                    conversation_history = step_obs.conversation_history

                results["step_logs"].append({
                    "step": step,
                    "agent_response": agent_response[:80] + "...",
                    "reward": reward,
                    "rule_score": rule_score,
                    "llm_score": llm_score,
                    "resolved": is_resolved,
                })

                if done:
                    results["resolved"] = is_resolved
                    print(f"\n  ✅ Episode ended at step {step} | Resolved: {is_resolved}")
                    break

            # --- SUMMARY ---
            results["steps"] = step
            results["total_reward"] = sum(log["reward"] for log in results["step_logs"])
            results["avg_reward"] = results["total_reward"] / step if step > 0 else 0.0
            results["avg_rule_score"] = sum(rule_scores) / len(rule_scores) if rule_scores else 0.0
            results["avg_llm_score"] = sum(llm_scores) / len(llm_scores) if llm_scores else 0.0
            
    except Exception as e:
        print(f"  [ERROR] Client session failed: {e}")

    return results



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 65)
    print("  BPO Customer Support Environment — Inference Runner")
    print("=" * 65)
    print(f"  Model   : {MODEL_NAME}")
    print(f"  Server  : {SERVER_URL}")
    print(f"  Tasks   : {', '.join(TASKS_TO_RUN)}")
    print("=" * 65)

    # Validate env vars
    if not OPENAI_API_KEY:
        print("\n[ERROR] OPENAI_API_KEY is not set. Please export your OpenRouter API key.")
        sys.exit(1)

    # Check server health
    print(f"\n⏳ Checking server at {SERVER_URL} ...")
    if not wait_for_server(SERVER_URL, timeout=10):
        print(f"[WARNING] Server at {SERVER_URL} not responding. Attempting to start it...")
        # Try to start server as a subprocess
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if not wait_for_server(SERVER_URL, timeout=20):
            print("[ERROR] Could not start server. Please run: uvicorn server.app:app --port 8000")
            sys.exit(1)
        print("✅ Server started successfully.")
    else:
        print("✅ Server is up!")

    # Run all tasks
    all_results = []
    for task_name in TASKS_TO_RUN:
        task_result = run_task(task_name)
        all_results.append(task_result)
        time.sleep(1)  # brief pause between tasks

    # Final Report
    print("\n" + "=" * 65)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 65)
    print(f"  {'Task':<25} {'Steps':>5} {'Avg Reward':>11} {'Rule':>8} {'LLM':>8} {'Resolved':>9}")
    print(f"  {'-'*25} {'-'*5} {'-'*11} {'-'*8} {'-'*8} {'-'*9}")

    overall_rewards = []
    for r in all_results:
        resolved_str = "✅ Yes" if r["resolved"] else "❌ No"
        print(
            f"  {r['task_name']:<25} {r['steps']:>5} "
            f"{r['avg_reward']:>11.4f} {r['avg_rule_score']:>8.4f} "
            f"{r['avg_llm_score']:>8.4f} {resolved_str:>9}"
        )
        overall_rewards.append(r["avg_reward"])

    if overall_rewards:
        print(f"\n  🏆 Overall Average Reward: {sum(overall_rewards)/len(overall_rewards):.4f}")

    print("\n  Done! All tasks completed.\n")


if __name__ == "__main__":
    main()
