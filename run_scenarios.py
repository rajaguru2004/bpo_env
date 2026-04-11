#!/usr/bin/env python3
"""
run_scenarios.py — Scenario-Based Test Runner for BPO OpenEnv Environment
=========================================================================

Executes predefined multi-step scenarios against the OpenEnv environment,
collects structured logs (reward, stage, mood, done, reward_reason, etc.),
and provides an enhanced summary report of performance.

Supports task-specific scenarios for order_status, damaged_product, and escalation.

Changes in v4:
- Enhanced summary table with Score/Stage/Status indicators.
- Display last step reward for partial scenarios (grader score is 0.0 unless done).
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path Initialization
# ---------------------------------------------------------------------------
_project_root = os.path.dirname(os.path.abspath(__file__))
_parent_root = os.path.dirname(_project_root)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _parent_root not in sys.path:
    sys.path.insert(0, _parent_root)

# Import local the native project client and models
try:
    # 1. Primary path: Unified bpo_env prefix satisfies IDE root
    from bpo_env.client import CustomerSupportEnv
    from bpo_env.models import CustomerSupportAction
except ImportError:
    try:
        # 2. Dynamic fallback: Hides from IDE linter but works at runtime
        import importlib
        _c_mod = importlib.import_module("client")
        _m_mod = importlib.import_module("models")
        CustomerSupportEnv = _c_mod.CustomerSupportEnv
        CustomerSupportAction = _m_mod.CustomerSupportAction
    except (ImportError, ModuleNotFoundError):
        CustomerSupportEnv = CustomerSupportAction = None # type: ignore

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL: str = "http://localhost:7860"
TASK_NAME: str = "order_status"
OUTPUT_FILE: str = "scenario_results.json"

# ---------------------------------------------------------------------------
# Scenario Definitions (Task-Specific Groups)
# ---------------------------------------------------------------------------

def get_order_status_scenarios() -> List[Dict[str, Any]]:
    return [
        {
            "name": "happy_path",
            "description": "Ideal flow: Empathy -> Tracking info -> Delivery Date -> Closure",
            "steps": [
                "I'm sorry for the delay. Let me check your order status for you.",
                "Your order #12345 has been shipped. The tracking number is TRK987654321 and it is currently in transit.",
                "Your order is expected to be delivered by April 3rd. Your reference number is REF-101. Is there anything else I can help you with?",
            ],
        },
        {
            "name": "incomplete_response",
            "description": "Agent missing tracking number in inquiry stage.",
            "steps": [
                "I'm sorry for the wait. Your order is on its way.",
            ],
        },
        {
            "name": "repetition",
            "description": "Agent repeating the same status update.",
            "steps": [
                "Your order #12345 is shipped. Tracking is TRK987654321.",
                "Your order #12345 is shipped. Tracking is TRK987654321.",
            ],
        },
    ]

def get_damaged_product_scenarios() -> List[Dict[str, Any]]:
    return [
        {
            "name": "happy_path_replacement",
            "description": "Ideal flow: Sincere Apology+Empathy -> Product Details Request -> Replacement Offer + Timeline -> Closure",
            "steps": [
                "I am so sorry to hear about the damaged item. I completely understand how frustrating this must be for you, but I'll make sure we fix it.",
                "Could you please confirm your order number so I can check the product details?",
                "I will arrange a replacement for you immediately. You should receive the new unit within 3-5 business days.",
                "Your reference number for this replacement is REF-456. Is there anything else I can assist with? Have a great day!",
            ],
        },
        {
            "name": "failed_empathy",
            "description": "Agent fails to show apology/empathy in start stage.",
            "steps": [
                "What is your order number? I can check on it.",
            ],
        },
        {
            "name": "recovery_flow",
            "description": "Agent starts poorly but recovers with a full replacement offer.",
            "steps": [
                "Please describe the damage.",
                "I am deeply sorry for the inconvenience and I hear your frustration. Let's make this right.",
                "I'll ship a replacement to you today. It will arrive in 48 hours.",
                "Thank you for your patience. Your case number is CASE-202.",
            ],
        },
    ]

def get_escalation_scenarios() -> List[Dict[str, Any]]:
    return [
        {
            "name": "happy_path_refund",
            "description": "Ideal flow: Deep Apology+Empathy -> Manager Escalation -> Full Refund + Timeline -> Closure",
            "steps": [
                "I sincerely apologize for the poor experience. I completely understand your frustration and I take full responsibility for this.",
                "I will escalate this immediately to my supervisor to oversee the resolution personally.",
                "We will process a full refund to your account within 48 hours as a gesture of goodwill.",
                "Your case number is CASE-888. Is there anything else I can assistance you with? Again, my apologies.",
            ],
        },
        {
            "name": "missing_manager_mention",
            "description": "Agent offers refund but fails to mention manager (required for acknowledgement stage).",
            "steps": [
                "I am so sorry. I understand.",
                "I will process your refund right now.",
            ],
        },
    ]

def get_scenarios(task_name: str) -> List[Dict[str, Any]]:
    if task_name == "damaged_product":
        base_suite = get_damaged_product_scenarios()
        common_context = "regarding your damaged product."
    elif task_name == "escalation":
        base_suite = get_escalation_scenarios()
        common_context = "about your escalation request."
    else:
        base_suite = get_order_status_scenarios()
        common_context = "regarding your order status."

    shared = [
        {
            "name": "irrelevant_response",
            "description": "Agent gives off-topic response.",
            "steps": [
                f"Please visit our website for more info {common_context}",
            ],
        },
        {
            "name": "stalling_agent",
            "description": "Agent asks for info repeatedly without progressing.",
            "steps": [
                "Could you please confirm your name?",
                "Could you also confirm your zip code?",
                "What was the city on the shipping address?",
            ],
        },
    ]
    
    return base_suite + shared


def get_stress_scenarios(task_name: str) -> List[Dict[str, Any]]:
    """
    Stress test scenarios: noisy inputs, incomplete queries, repeated user prompts.
    Validates robustness: no collapse, no infinite loops, recovery works.
    """
    common_stress = [
        {
            "name": "stress_triple_repeat",
            "description": "[STRESS] User asks same thing 2 times then fixes; agent must not repeat",
            "steps": [
                "Where is my order? 12345",
                "WHERE IS MY ORDER? 12345",
                "I sincerely apologize. I am processing your request immediately. Your order #12345 is in transit (TRK987654321) and will arrive April 3rd.",
            ],
        },
        {
            "name": "stress_contradiction",
            "description": "[STRESS] User changes mind mid-flow",
            "steps": [
                "I want a refund.",
                "Wait, no, just send a replacement instead.",
                "I completely understand. I'll cancel the refund and ship a replacement unit to you today instead.",
            ],
        },
        {
            "name": "stress_mood_swing",
            "description": "[STRESS] User goes from neutral to angry",
            "steps": [
                "Hi, I have an issue with my order.",
                "I AM TIRED OF WAITING. FIX THIS NOW OR I'LL CALL MY BANK!!",
                "I sincerely apologize for the delay and I hear your frustration. I am taking immediate action to resolve this for you.",
            ],
        }
    ]

    if task_name == "order_status":
        base = [
            {
                "name": "stress_noisy_input",
                "description": "[STRESS] Agent handles garbled/noisy user message",
                "steps": [
                    "Uh... my order? Number 12345? It's just... I don't know where it is???",
                    "Your order #12345 has been shipped with tracking number TRK987654321. It is currently in transit.",
                    "Your order is expected to be delivered by April 3rd. Your reference number is REF-777. Is there anything else I can help with?",
                ],
            },
            {
                "name": "stress_incomplete_query",
                "description": "[STRESS] User sends extremely short/incomplete message",
                "steps": [
                    "Order?",
                    "Your order #12345 has been shipped. Tracking number: TRK987654321, currently in transit.",
                    "Expected delivery: April 3rd. Case number REF-888. Is there anything else?",
                ],
            },
            {
                "name": "stress_repeated_user_prompt",
                "description": "[STRESS] User repeats same complaint; agent must not loop",
                "steps": [
                    "I need my order status. Order 12345.",
                    "Hello! Let me help you with order 12345.",
                    "Your order is shipped — TRK987654321. Expected to arrive April 3rd. Reference: REF-999.",
                ],
            },
        ]
        return base + common_stress

    elif task_name == "damaged_product":
        base = [
            {
                "name": "stress_noisy_input",
                "description": "[STRESS] Noisy/all-caps angry complaint",
                "steps": [
                    "I am truly sorry to hear your product arrived damaged. I completely understand your frustration.",
                    "Could you please confirm your order number so I can check the details?",
                    "I will arrange a replacement immediately — you'll receive the new unit within 3-5 business days.",
                    "Your case number is CASE-STRESS-1. Is there anything else I can assist with?",
                ],
            },
            {
                "name": "stress_incomplete_query",
                "description": "[STRESS] User only says 'damaged' — agent must probe then resolve",
                "steps": [
                    "Damaged.",
                    "I sincerely apologize and I completely understand your frustration. Let me help you right away.",
                    "Could you confirm your order number so I can get the replacement started?",
                    "I will ship a replacement within 3-5 business days. Case: CASE-STRESS-2. Anything else?",
                ],
            },
            {
                "name": "stress_repeated_user_prompt",
                "description": "[STRESS] User repeats complaint; agent must not stall",
                "steps": [
                    "I'm so sorry to hear about the damage. I completely understand and I'll make this right.",
                    "I received a damaged product. I want a replacement.",
                    "I will arrange a replacement for your Bluetooth Speaker right away — expected in 3-5 business days.",
                    "Your reference number is REF-STRESS-3. Have a great day!",
                ],
            },
        ]
        return base + common_stress

    else:  # escalation
        base = [
            {
                "name": "stress_noisy_input",
                "description": "[STRESS] Incoherent angry message",
                "steps": [
                    "I sincerely apologize for this terrible experience. I completely understand your frustration and I take full responsibility.",
                    "I am escalating this directly to our senior manager who will personally oversee your case.",
                    "We will process a full refund within 3-5 business days. I assure you this will be prioritised.",
                    "Your case number is CASE-ESC-STRESS. Is there anything else I can do? Again, I sincerely apologize.",
                ],
            },
            {
                "name": "stress_incomplete_query",
                "description": "[STRESS] Customer only says 'refund now' — agent must handle",
                "steps": [
                    "Refund. Now.",
                    "I sincerely apologize and I completely understand your frustration. I will process your full refund immediately.",
                    "I am escalating this to a senior supervisor as well — your refund will be processed within 3-5 business days.",
                    "Case number: CASE-ESC-S2. Thank you for your patience. Is there anything else I can help with?",
                ],
            },
            {
                "name": "stress_repeated_user_prompt",
                "description": "[STRESS] User repeats refund demand; agent must not loop",
                "steps": [
                    "I sincerely apologize. I take full responsibility for this awful experience.",
                    "I want a full refund and a manager RIGHT NOW.",
                    "I am escalating to a senior manager immediately and processing your full refund within 48 hours.",
                    "Case ID: CASE-ESC-S3. Refund confirmed. Is there anything else I can assist you with today?",
                ],
            },
        ]
        return base + common_stress


# ---------------------------------------------------------------------------
# Data Models and Extraction
# ---------------------------------------------------------------------------

STEP_FIELDS = [
    "reward", "done", "conversation_stage", "customer_mood",
    "reward_reason", "rule_score", "grader_score",
    "repetition_count", "stall_count", "intents_detected", "intents",
    "is_resolved", "success", "failure_reason"
]

def extract_observation_data(obs: Any, reward: float, done: bool, action: str, step_num: int) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "step": step_num,
        "action": action,
        "reward": reward,
        "done": done,
    }
    for field in STEP_FIELDS:
        if field in ["reward", "done"]: continue
        val = getattr(obs, field, None)
        data[field] = val
    return data

# ---------------------------------------------------------------------------
# Logging and Output Helpers
# ---------------------------------------------------------------------------

SEPARATOR = "─" * 65

def print_scenario_header(name: str, description: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"[SCENARIO] {name}")
    if description:
        print(f"  Description: {description}")
    print(SEPARATOR)

def print_step_log(step_data: Dict[str, Any]) -> None:
    step_num = step_data.get("step", "?")
    action = step_data.get("action", "")
    reward = step_data.get("reward")
    stage = step_data.get("conversation_stage")
    mood = step_data.get("customer_mood")
    done = step_data.get("done")
    intents = step_data.get("intents_detected", "")
    repetition = step_data.get("repetition_count")
    stall = step_data.get("stall_count")
    grader = step_data.get("grader_score")

    print(f"\nStep {step_num}:")
    print(f"  Action        : {action if len(str(action)) <= 90 else str(action)[:87] + '...'}")
    print(f"  Reward        : {reward:.4f}" if isinstance(reward, float) else f"  Reward        : {reward}")
    print(f"  Stage         : {stage}")
    print(f"  Mood          : {mood}")
    print(f"  Intents       : {', '.join(map(str, intents)) if isinstance(intents, list) else intents}")
    
    # Display the new dictionary-based intents (v7 patch)
    if "intents" in step_data and isinstance(step_data["intents"], dict):
        active = [k for k, v in step_data["intents"].items() if v]
        if active:
            print(f"  Active Intents: {', '.join(active)}")
    if repetition: print(f"  Repetitions   : {repetition}")
    if stall: print(f"  Stall Count   : {stall}")
    if done and grader is not None:
        print(f"  Grader Score  : {grader:.4f}  ← final episode score")


# ---------------------------------------------------------------------------
# Scenario Execution Logic
# ---------------------------------------------------------------------------

def run_scenario(env: Any, scenario: Dict[str, Any], task_name: str) -> Dict[str, Any]:
    name = scenario["name"]
    steps_config = scenario["steps"]

    print_scenario_header(name, scenario.get("description", ""))

    try:
        reset_result = env.reset(task_name=task_name)
        print(f"  Reset OK — task: {task_name}")
    except Exception as exc:
        print(f"  [ERROR] Reset failed: {exc}")
        return {"name": name, "error": str(exc)}

    collected_steps: List[Dict[str, Any]] = []

    for idx, action_text in enumerate(steps_config, start=1):
        try:
            action = CustomerSupportAction(response=action_text)
            result = env.step(action)
            step_data = extract_observation_data(result.observation, result.reward or 0.0, result.done, action_text, idx)
            collected_steps.append(step_data)
            print_step_log(step_data)
            if result.done:
                print(f"\n  ✓ Episode ended at step {idx}")
                break
            time.sleep(0.01)
        except Exception as exc:
            print(f"  [ERROR] Step {idx} failed: {exc}")
            break

    if not collected_steps:
        return {"name": name, "error": "No steps completed"}

    final = collected_steps[-1]
    return {
        "name": name,
        "task_name": task_name,
        "steps": collected_steps,
        "total_steps": len(collected_steps),
        "final_done": final.get("done"),
        "final_reward": final.get("reward"),
        "final_grader_score": final.get("grader_score"),
        "final_stage": final.get("conversation_stage"),
        "final_mood": final.get("customer_mood"),
        "is_resolved": final.get("is_resolved", False),
    }

# ---------------------------------------------------------------------------
# Main Summary Table Logic
# ---------------------------------------------------------------------------

def display_summary(results: List[Dict[str, Any]]) -> None:
    print("\n\n" + "═" * 78)
    print("  STRUCTURED PERFORMANCE SUMMARY")
    print("  (Note: Scores for partial runs reflect the reward at that step)")
    print("═" * 78)
    
    # Header
    print(f"{'Scenario':<30} | {'Status':<12} | {'Steps':<5} | {'Score/Reward':<12} | {'Final Stage'}")
    print("─" * 78)

    for res in results:
        if "error" in res:
            print(f"{res['name']:<30} | {'❌ Error':<12} | {0:<5} | {'0.000':<12} | ---")
            continue

        done = res.get("final_done", False)
        resolved = res.get("is_resolved", False)
        steps = res.get("total_steps", 0)
        stage = res.get("final_stage", "start")
        
        # Display Logic: Use Grader Score if done, else the final step reward
        grader = res.get("final_grader_score")
        reward = res.get("final_reward")
        
        display_score = grader if (done and grader and grader > 0) else reward
        if display_score is None: display_score = 0.0

        # Status logic
        if done:
            status = "DONE (✓)" if resolved else "FAIL (❌)"
        else:
            status = "PARTIAL (⋯)"

        name_trunc = res["name"] if len(res["name"]) <= 30 else res["name"][:27] + "..."
        print(f"{name_trunc:<30} | {status:<12} | {steps:<5} | {display_score:<12.3f} | {stage}")

    print("═" * 78)


def run_stress_scenarios(base_url: str, task_name: str, output_file: str) -> None:
    """Run stress test scenarios and validate robustness criteria."""
    print("\n" + "═" * 78)
    print(f"  BPO STRESS TEST — Task: {task_name}")
    print(f"  Server  : {base_url} | Time: {datetime.now().strftime('%H:%M:%S')}")
    print("  Validates: no collapse, no infinite loops, recovery within 2 steps")
    print("═" * 78)

    scenarios = get_stress_scenarios(task_name)
    results: List[Dict[str, Any]] = []
    validation_failures: List[str] = []

    client_env = CustomerSupportEnv(base_url=base_url).sync()
    with client_env as env:
        for scen in scenarios:
            res = run_scenario(env, scen, task_name)
            results.append(res)

            # ── Stress validation checks ──────────────────────────────────────
            name = scen["name"]
            steps_data = res.get("steps", [])

            # Check 1: No collapse (all steps should return non-zero reward)
            zero_reward_steps = [
                s["step"] for s in steps_data
                if isinstance(s.get("reward"), (int, float)) and s["reward"] < 0.01
            ]
            if zero_reward_steps:
                validation_failures.append(
                    f"{name}: COLLAPSE detected at steps {zero_reward_steps}"
                )

            # Check 2: No infinite loop (stall_count must not exceed 4 in any step)
            high_stall = [
                s["step"] for s in steps_data
                if isinstance(s.get("stall_count"), int) and s["stall_count"] > 4
            ]
            if high_stall:
                validation_failures.append(
                    f"{name}: INFINITE LOOP risk — stall_count > 4 at steps {high_stall}"
                )

            # Check 3: Recovery within 2 steps (after a low-reward step)
            for i, step_d in enumerate(steps_data[:-1]):
                if isinstance(step_d.get("reward"), (int, float)) and step_d["reward"] < 0.3:
                    recovery_window = steps_data[i+1:i+3]
                    recovered = any(
                        isinstance(s.get("reward"), (int, float)) and s["reward"] >= 0.4
                        for s in recovery_window
                    )
                    if not recovered and recovery_window:
                        validation_failures.append(
                            f"{name}: RECOVERY FAILED after low-reward at step {step_d['step']}"
                        )
                    break  # only check first low-reward occurrence

    display_summary(results)

    print("\n" + "═" * 78)
    print("  STRESS VALIDATION RESULTS")
    print("═" * 78)
    if not validation_failures:
        print("  ✅ ALL STRESS CHECKS PASSED — No collapse, loops, or recovery failures")
    else:
        print(f"  ❌ {len(validation_failures)} VALIDATION FAILURE(S):")
        for fail in validation_failures:
            print(f"     • {fail}")
    print("═" * 78)

    stress_output = output_file.replace(".json", "_stress.json")
    try:
        with open(stress_output, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "meta": {"task_name": task_name, "mode": "stress"},
                    "results": results,
                    "validation_failures": validation_failures,
                },
                fh, indent=2, default=str
            )
        print(f"\n✓ Stress results saved to: {stress_output}")
    except Exception as exc:
        print(f"\n[WARNING] Could not save stress output: {exc}")


def run_all_scenarios(base_url: str, task_name: str, output_file: str) -> None:
    print("\n" + "═" * 78)
    print(f"  BPO Test Runner Multi-Task v4 — Task: {task_name}")
    print(f"  Server  : {base_url} | Time: {datetime.now().strftime('%H:%M:%S')}")
    print("═" * 78)

    scenarios = get_scenarios(task_name)
    results: List[Dict[str, Any]] = []

    client_env = CustomerSupportEnv(base_url=base_url).sync()
    with client_env as env:
        for scen in scenarios:
            results.append(run_scenario(env, scen, task_name))

    display_summary(results)

    try:
        with open(output_file, "w", encoding="utf-8") as fh:
            json.dump({"meta": {"task_name": task_name}, "results": results}, fh, indent=2, default=str)
        print(f"\n✓ Full results saved to: {output_file}")
    except Exception as exc:
        print(f"\n[WARNING] Could not save output: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BPO Scenario Test Runner")
    parser.add_argument("--url", default=BASE_URL, help="Environment server URL")
    parser.add_argument(
        "--task", default=TASK_NAME,
        choices=["order_status", "damaged_product", "escalation"],
        help="Task to run scenarios for",
    )
    parser.add_argument("--output", default=OUTPUT_FILE, help="Output JSON file path")
    parser.add_argument(
        "--stress", action="store_true",
        help="Run stress test scenarios (noisy inputs, incomplete queries, repeated prompts)",
    )
    parser.add_argument(
        "--all-tasks", action="store_true",
        help="Run scenarios for all 3 tasks sequentially",
    )
    args = parser.parse_args()

    url = args.url.rstrip("/")

    if args.all_tasks:
        tasks = ["order_status", "damaged_product", "escalation"]
        for t in tasks:
            out = args.output.replace(".json", f"_{t}.json")
            if args.stress:
                run_stress_scenarios(url, t, out)
            else:
                run_all_scenarios(url, t, out)
    else:
        if args.stress:
            run_stress_scenarios(url, args.task, args.output)
        else:
            run_all_scenarios(url, args.task, args.output)
