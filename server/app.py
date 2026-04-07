"""
FastAPI application for the BPO Customer Support Environment.

Endpoints:
    - POST /reset          Reset environment with optional task_name
    - POST /step           Execute an agent action
    - GET  /state          Get current environment state
    - GET  /schema         Get action/observation schemas
    - GET  /tasks          List all tasks with grader metadata (Phase 2 validation)
    - POST /grade          Grade a completed trajectory (task_name + trajectory body)
    - POST /grade/{task}   Per-task grader for a single response (Phase 2 runtime)
    - WS   /ws             WebSocket endpoint for persistent sessions

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

import sys
import os
from typing import Any
from fastapi import Request

# ---------------------------------------------------------------------------
# Path setup — ensure project root is importable
# ---------------------------------------------------------------------------
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# ---------------------------------------------------------------------------
# OpenEnv core server
# ---------------------------------------------------------------------------
try:
    from openenv.core.env_server.http_server import create_app
except Exception as exc:
    raise ImportError(
        "openenv is required. Install dependencies with:\n    uv sync\n"
    ) from exc

# ---------------------------------------------------------------------------
# Environment models and implementation
# ---------------------------------------------------------------------------
try:
    from models import CustomerSupportAction, CustomerSupportObservation
    from server.bpo_env_environment import CustomerSupportEnvironment
except ImportError:
    from ..models import CustomerSupportAction, CustomerSupportObservation
    from .bpo_env_environment import CustomerSupportEnvironment

# ---------------------------------------------------------------------------
# Graders — single source of truth: tasks.py
# These are plain callables with signature (response: str, state: dict) -> float
# ---------------------------------------------------------------------------
from tasks import (
    grade_episode,
    grade_order_status,
    grade_damaged_product,
    grade_escalation,
)

# ---------------------------------------------------------------------------
# FastAPI app (created by openenv helper)
# ---------------------------------------------------------------------------
app = create_app(
    CustomerSupportEnvironment,
    CustomerSupportAction,
    CustomerSupportObservation,
    env_name="bpo_env",
    max_concurrent_envs=3,
)

# ---------------------------------------------------------------------------
# Per-task environment subclasses (for task-specific resets)
# ---------------------------------------------------------------------------

class OrderStatusEnv(CustomerSupportEnvironment):
    def reset(self, **kwargs):
        return super().reset(task_name="order_status", **kwargs)


class DamagedProductEnv(CustomerSupportEnvironment):
    def reset(self, **kwargs):
        return super().reset(task_name="damaged_product", **kwargs)


class EscalationEnv(CustomerSupportEnvironment):
    def reset(self, **kwargs):
        return super().reset(task_name="escalation", **kwargs)


# ---------------------------------------------------------------------------
# Task registry — maps task name → env class + grader function
# The grader must be a plain callable: (response: str, state: dict) -> float
# ---------------------------------------------------------------------------
TASK_REGISTRY = {
    "order_status": {
        "env_class": OrderStatusEnv,
        "grader": grade_order_status,       # function from tasks.py
    },
    "damaged_product": {
        "env_class": DamagedProductEnv,
        "grader": grade_damaged_product,    # function from tasks.py
    },
    "escalation": {
        "env_class": EscalationEnv,
        "grader": grade_escalation,         # function from tasks.py
    },
}

# ---------------------------------------------------------------------------
# /tasks — Phase 2 deep validation endpoint
# ---------------------------------------------------------------------------

@app.get("/tasks")
async def list_tasks():
    """
    Return all tasks with grader metadata.
    Phase 2 deep validation uses this to confirm ≥3 tasks have graders.
    """
    return {
        "tasks": [
            {
                "name": "order_status",
                "difficulty": "easy",
                "description": "Customer wants to know the status of their order.",
                "has_grader": True,
                "grader": "grade_order_status",
                "grader_module": "tasks",
                "enabled": True,
            },
            {
                "name": "damaged_product",
                "difficulty": "medium",
                "description": "Customer received a damaged product and wants a replacement or refund.",
                "has_grader": True,
                "grader": "grade_damaged_product",
                "grader_module": "tasks",
                "enabled": True,
            },
            {
                "name": "escalation",
                "difficulty": "hard",
                "description": "Angry customer demanding a full refund and to speak with a manager.",
                "has_grader": True,
                "grader": "grade_escalation",
                "grader_module": "tasks",
                "enabled": True,
            },
        ],
        "total": 3,
        "graded_count": 3,
    }


# ---------------------------------------------------------------------------
# /grade — Grade a full trajectory
# ---------------------------------------------------------------------------

@app.post("/grade")
async def grade_trajectory(request: Request):
    """
    Grade a completed trajectory for the given task.
    Body: {"task_name": str, "trajectory": list | dict}
    Returns: {"score": float}
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    task_name  = body.get("task_name", "order_status")
    trajectory = body.get("trajectory", {})
    score = grade_episode(task_name, trajectory)
    # Clamp to (0.01, 0.99) — validator rejects exactly 0.0 and 1.0
    score = max(0.01, min(0.99, float(score)))
    return {"task": task_name, "score": score}


# ---------------------------------------------------------------------------
# /grade/{task_name} — Per-task grader for single response (Phase 2 runtime)
# ---------------------------------------------------------------------------

@app.post("/grade/{task_name}")
async def grade_task(task_name: str, request: Request):
    """
    Per-task grader endpoint for Phase 2 runtime validation.
    Body: {"response": str, "state": dict}
    Returns: {"task": str, "score": float, "reward": float, "done": bool}
    """
    task = TASK_REGISTRY.get(task_name)
    if not task:
        return {
            "error": f"Unknown task: {task_name}",
            "score": 0.01,
            "reward": 0.01,
            "done": True,
        }
    try:
        body = await request.json()
    except Exception:
        body = {}

    response = body.get("response", "")
    state    = body.get("state", {})

    # grader is a plain callable: (response: str, state: dict) -> float
    score = float(task["grader"](response, state))
    # Clamp to (0.01, 0.99)
    score = max(0.01, min(0.99, score))

    return {
        "task":   task_name,
        "score":  round(score, 4),
        "reward": round(score, 4),
        "done":   True,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Entry point for direct execution via uv run or python -m."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
