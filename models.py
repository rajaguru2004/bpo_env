"""
Data models for the BPO Customer Support Environment.

The bpo_env environment simulates real-world customer support conversations
where an LLM agent acts as a customer support executive.

Version 2: Extended with stateful multi-step fields — stage, mood,
issue_status, intent, and hints.
"""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class CustomerSupportAction(Action):
    """Action for the Customer Support environment — the agent's response to the customer."""

    response: str = Field(..., description="The agent's response to the customer message")


class CustomerSupportObservation(Observation):
    """
    Observation from the Customer Support environment.

    Contains both the raw conversation state and rich metadata about the
    current episode stage, customer mood, and intermediate rewards for
    transparent multi-step learning.
    """

    # ── Core conversation ──────────────────────────────────────────────────
    customer_message: str = Field(
        default="",
        description="The latest message from the customer",
    )
    conversation_history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Full conversation history as list of {role, content} dicts",
    )

    # ── Task identity ──────────────────────────────────────────────────────
    task_name: str = Field(
        default="",
        description="Current task identifier (order_status, damaged_product, escalation)",
    )
    task_difficulty: str = Field(
        default="easy",
        description="Task difficulty: easy, medium, or hard",
    )
    task_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional task-specific context information (order IDs, etc.)",
    )

    # ── Episode progress ───────────────────────────────────────────────────
    step: int = Field(default=0, description="Current step number in the episode")
    max_steps: int = Field(default=10, description="Maximum steps allowed for this episode")
    is_resolved: bool = Field(default=False, description="Whether the customer issue has been resolved")

    # ── State machine fields (NEW) ─────────────────────────────────────────
    conversation_stage: str = Field(
        default="start",
        description="Current stage of the conversation (start,empathy,diagnosis,resolution,closure)",
    )
    customer_mood: str = Field(
        default="neutral",
        description="Current customer mood: angry | neutral | satisfied",
    )
    issue_status: str = Field(
        default="unresolved",
        description="Issue lifecycle status: unresolved | in_progress | resolved",
    )
    intent_detected: str = Field(
        default="",
        description="Classified intent of the agent's last response",
    )
    hints: List[str] = Field(
        default_factory=list,
        description="Optional guidance hints (populated on easy tasks to aid learning)",
    )

    # ── Reward components ──────────────────────────────────────────────────
    rule_score: float = Field(
        default=0.0,
        description="Rule-based component of reward (0.0–1.0)",
    )
    llm_score: float = Field(
        default=0.0,
        description="LLM judge component of reward (0.0–1.0, only at episode end)",
    )
    stage_reward: float = Field(
        default=0.0,
        description="Bonus reward for advancing conversation stage",
    )
    final_reward: float = Field(
        default=0.0,
        description="Combined final reward (0.0–1.0)",
    )
    grader_score: float = Field(
        default=0.0,
        description="Deterministic grader score (0.0–1.0), populated only at done=True",
    )
