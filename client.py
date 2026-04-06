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
            # ── Core conversation ──────────────────────────────────────────
            customer_message=obs_data.get("customer_message", ""),
            conversation_history=obs_data.get("conversation_history", []),
            # ── Task identity ──────────────────────────────────────────────
            task_name=obs_data.get("task_name", ""),
            task_difficulty=obs_data.get("task_difficulty", "easy"),
            task_context=obs_data.get("task_context"),
            # ── Episode progress ───────────────────────────────────────────
            step=obs_data.get("step", 0),
            max_steps=obs_data.get("max_steps", 10),
            is_resolved=obs_data.get("is_resolved", False),
            # ── State machine fields (v7 Patch) ────────────────────────────
            conversation_stage=obs_data.get("conversation_stage", "start"),
            customer_mood=obs_data.get("customer_mood", "neutral"),
            issue_status=obs_data.get("issue_status", "unresolved"),
            intent_detected=obs_data.get("intent_detected", ""),
            intents_detected=obs_data.get("intents_detected", []),
            intents=obs_data.get("intents", {}),
            hints=obs_data.get("hints", []),
            # ── Diagnostics ────────────────────────────────────────────────
            success=obs_data.get("success", False),
            repetition_count=obs_data.get("repetition_count", 0),
            stall_count=obs_data.get("stall_count", 0),
            failure_reason=obs_data.get("failure_reason", ""),
            # ── Reward components ──────────────────────────────────────────
            rule_score=obs_data.get("rule_score", 0.0),
            llm_score=obs_data.get("llm_score", 0.0),
            reward_reason=obs_data.get("reward_reason", ""),
            grader_score=obs_data.get("grader_score", 0.0),
            # ── OpenEnv standard ───────────────────────────────────────────
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_reset_result(self, payload: Dict) -> StepResult[CustomerSupportObservation]:
        """Parse server reset response."""
        return self._parse_result(payload)

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
