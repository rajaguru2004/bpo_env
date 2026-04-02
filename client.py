"""Customer Support Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from models import CustomerSupportAction, CustomerSupportObservation
except ImportError:
    from .models import CustomerSupportAction, CustomerSupportObservation



class CustomerSupportEnv(
    EnvClient[CustomerSupportAction, CustomerSupportObservation, State]
):
    """
    Client for the BPO Customer Support Environment.

    Maintains a persistent WebSocket connection to the environment server,
    enabling multi-step interactions with lower latency.

    Example:
        >>> env = CustomerSupportEnv(base_url="http://localhost:8000")
        >>> result = env.reset(task_name="order_status")
        >>> print(result.observation.customer_message)
        >>>
        >>> result = env.step(CustomerSupportAction(
        ...     response="I'd be happy to help! Your order #12345 has been shipped."
        ... ))
        >>> print(f"Reward: {result.reward}, Done: {result.done}")
        >>> env.close()
    """

    def _step_payload(self, action: CustomerSupportAction) -> Dict:
        """Convert CustomerSupportAction to JSON payload."""
        return {"response": action.response}

    def _parse_result(self, payload: Dict) -> StepResult[CustomerSupportObservation]:
        """Parse server response into StepResult[CustomerSupportObservation]."""
        # In WebSocket mode, the payload itself contains the serialized observation nested under "observation"
        obs_data = payload.get("observation", {})
        observation = CustomerSupportObservation(
            customer_message=obs_data.get("customer_message", ""),
            conversation_history=obs_data.get("conversation_history", []),
            task_name=obs_data.get("task_name", ""),
            task_difficulty=obs_data.get("task_difficulty", "easy"),
            step=obs_data.get("step", 0),
            max_steps=obs_data.get("max_steps", 10),
            is_resolved=obs_data.get("is_resolved", False),
            rule_score=obs_data.get("rule_score", 0.0),
            llm_score=obs_data.get("llm_score", 0.0),
            final_reward=obs_data.get("final_reward", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
            task_context=obs_data.get("task_context"),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
