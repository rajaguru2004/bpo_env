from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import sys
import os

# Ensure the server directory is in sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "server"))

try:
    from server.grader import grade_episode as internal_grade_episode
except ImportError:
    # Fallback if already in sys.path
    from grader import grade_episode as internal_grade_episode

@dataclass(frozen=True)
class TaskConfig:
    name: str
    difficulty: str
    description: str
    max_steps: int

TASK_CONFIGS: Dict[str, TaskConfig] = {
    "order_status": TaskConfig(
        name="order_status",
        difficulty="easy",
        max_steps=5,
        description="Customer wants to know the status of their order.",
    ),
    "damaged_product": TaskConfig(
        name="damaged_product",
        difficulty="medium",
        max_steps=8,
        description="Customer received a damaged product and wants a replacement or refund.",
    ),
    "escalation": TaskConfig(
        name="escalation",
        difficulty="hard",
        max_steps=12,
        description="Angry customer demanding a full refund and to speak with a manager.",
    ),
}

def grade_episode(task_name: str, trajectory: Any) -> float:
    """
    OpenEnv-compliant grader entry point.
    
    If trajectory is a list, it's treated as a step-by-step history.
    If it's a dict, it's treated as a pre-computed episode summary.
    """
    # If it's a dict and contains the necessary keys, it might be from a direct call
    # In BPO Env, we typically need the full trajectory for quality scoring.
    if isinstance(trajectory, dict):
        # Extract fields for internal_grade_episode
        # Note: BPO Env business logic requires a list of steps for full quality analysis.
        # If we only have a summary, we provide a placeholder trajectory.
        steps = trajectory.get("trajectory", [])
        if not steps and "steps" in trajectory:
            steps = trajectory["steps"]
            
        final_stage = trajectory.get("final_stage", "closure")
        final_mood = trajectory.get("final_mood", "neutral")
        resolved = trajectory.get("resolved", True)
        closure_reached = trajectory.get("closure_reached", True)
        steps_taken = trajectory.get("steps_taken", len(steps))
        max_steps = trajectory.get("max_steps", TASK_CONFIGS.get(task_name).max_steps if task_name in TASK_CONFIGS else 10)
        step_rewards = trajectory.get("step_rewards", [0.0] * steps_taken)
        required_intents = trajectory.get("required_intents", [])
        
        return internal_grade_episode(
            trajectory=steps,
            final_stage=final_stage,
            final_mood=final_mood,
            resolved=resolved,
            closure_reached=closure_reached,
            steps_taken=steps_taken,
            max_steps=max_steps,
            step_rewards=step_rewards,
            required_intents=required_intents
        )
    
    # If trajectory is already a list, assume it's the full history
    if isinstance(trajectory, list):
        if not trajectory:
            return 0.0
            
        final_step = trajectory[-1]
        # In BPO Env, the last step observation contains redundant summary info
        # See CustomerSupportEnvironment.step() in bpo_env_environment.py
        
        return internal_grade_episode(
            trajectory=trajectory,
            final_stage=final_step.get("conversation_stage", "closure"),
            final_mood=final_step.get("customer_mood", "neutral"),
            resolved=final_step.get("is_resolved", True),
            closure_reached=final_step.get("success", True),
            steps_taken=len(trajectory),
            max_steps=final_step.get("max_steps", 10),
            step_rewards=[s.get("reward", 0.0) for s in trajectory],
            required_intents=[] # Can be populated if needed
        )

    return 0.0
