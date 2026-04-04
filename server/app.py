"""
FastAPI application for the BPO Customer Support Environment.

Endpoints:
    - POST /reset: Reset environment with optional task_name
    - POST /step: Execute an agent action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required. Install dependencies with '\n    uv sync\n'"
    ) from e

import sys
import os

# Ensure the root directory is in sys.path for absolute imports
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

try:
    from models import CustomerSupportAction, CustomerSupportObservation
    from server.bpo_env_environment import CustomerSupportEnvironment
except ImportError:
    from ..models import CustomerSupportAction, CustomerSupportObservation
    from .bpo_env_environment import CustomerSupportEnvironment


app = create_app(
    CustomerSupportEnvironment,
    CustomerSupportAction,
    CustomerSupportObservation,
    env_name="bpo_env",
    max_concurrent_envs=3,
)


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
