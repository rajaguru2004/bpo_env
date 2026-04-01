"""
Data models for the BPO Customer Support Environment.

The bpo_env environment simulates real-world customer support conversations
where an LLM agent acts as a customer support executive.
"""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class CustomerSupportAction(Action):
    """Action for the Customer Support environment — the agent's response to the customer."""

    response: str = Field(..., description="The agent's response to the customer message")


class CustomerSupportObservation(Observation):
    """Observation from the Customer Support environment."""

    customer_message: str = Field(
        default="",
        description="The latest message from the customer",
    )
    conversation_history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Full conversation history as list of {role, content} dicts",
    )
    task_name: str = Field(
        default="",
        description="Current task identifier (order_status, damaged_product, escalation)",
    )
    task_difficulty: str = Field(
        default="easy",
        description="Task difficulty: easy, medium, or hard",
    )
    step: int = Field(
        default=0,
        description="Current step number in the episode",
    )
    max_steps: int = Field(
        default=10,
        description="Maximum steps allowed for this episode",
    )
    is_resolved: bool = Field(
        default=False,
        description="Whether the customer issue has been resolved",
    )
    rule_score: float = Field(
        default=0.0,
        description="Rule-based component of reward (0.0–1.0)",
    )
    llm_score: float = Field(
        default=0.0,
        description="LLM judge component of reward (0.0–1.0)",
    )
    final_reward: float = Field(
        default=0.0,
        description="Combined final reward (0.0–1.0)",
    )
    task_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional task-specific context information",
    )
