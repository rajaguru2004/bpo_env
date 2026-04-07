"""
FastAPI application for the BPO Customer Support Environment.

Endpoints:
    - POST /reset: Reset environment with optional task_name
    - POST /step: Execute an agent action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 7860
"""

import sys
import os
from typing import Any

# Ensure the root directory is in sys.path for absolute imports
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from models import CustomerSupportAction, CustomerSupportObservation
    from server.bpo_env_environment import CustomerSupportEnvironment
    from server.graders import (
        OrderStatusGrader,
        DamagedProductGrader,
        EscalationGrader,
    )
except ImportError:
    from ..models import CustomerSupportAction, CustomerSupportObservation
    from .bpo_env_environment import CustomerSupportEnvironment
    from .graders import (
        OrderStatusGrader,
        DamagedProductGrader,
        EscalationGrader,
    )

# Import grader from the root-level tasks module (self-contained, no circular deps)
from tasks import grade_episode


app = create_app(
    CustomerSupportEnvironment,
    CustomerSupportAction,
    CustomerSupportObservation,
    env_name="bpo_env",
    max_concurrent_envs=3,
)


# ---------------------------------------------------------------------------
# Task Registry & Discovery
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


TASK_REGISTRY = {
    "order_status": {
        "env_class": OrderStatusEnv,
        "grader": OrderStatusGrader(),
    },
    "damaged_product": {
        "env_class": DamagedProductEnv,
        "grader": DamagedProductGrader(),
    },
    "escalation": {
        "env_class": EscalationEnv,
        "grader": EscalationGrader(),
    },
}


@app.post("/grade")
async def grade_trajectory(task_name: str, trajectory: Any):
    """Grade a completed trajectory for the given task."""
    return {"score": grade_episode(task_name, trajectory)}


@app.get("/tasks")
async def list_tasks():
    """
    Return all tasks with their grader information.
    Phase 2 deep validation uses this endpoint to verify that
    at least 3 tasks have graders attached.
    """
    return {
        "tasks": [
            {
                "name": "order_status",
                "difficulty": "easy",
                "description": "Customer wants to know the status of their order.",
                "has_grader": True,
                "grader": "grade_episode",
                "grader_module": "tasks",
            },
            {
                "name": "damaged_product",
                "difficulty": "medium",
                "description": "Customer received a damaged product and wants a replacement or refund.",
                "has_grader": True,
                "grader": "grade_episode",
                "grader_module": "tasks",
            },
            {
                "name": "escalation",
                "difficulty": "hard",
                "description": "Angry customer demanding a full refund and to speak with a manager.",
                "has_grader": True,
                "grader": "grade_episode",
                "grader_module": "tasks",
            },
        ],
        "total": 3,
        "graded_count": 3,
    }


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
