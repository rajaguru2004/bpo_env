"""
Data models for the BPO Customer Support Environment.

The bpo_env environment simulates real-world customer support conversations
where an LLM agent acts as a customer support executive.

Version 5: Normalized reward system — deterministic, pure rule-based grader,
bounded reward [0,1], reward equals grader_score at episode end.
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

    # ── State machine fields ───────────────────────────────────────────────
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
        description="Primary classified intent of the agent's last response (first match, backwards-compat)",
    )
    intents_detected: List[str] = Field(
        default_factory=list,
        description="All intents detected in the agent's last response (multi-intent v4)",
    )
    intents: Dict[str, bool] = Field(
        default_factory=dict,
        description="Detailed intent flags (v7 patch)",
    )
    hints: List[str] = Field(
        default_factory=list,
        description="Optional guidance hints (populated on easy tasks to aid learning)",
    )

    # ── State machine diagnostics ──────────────────────────────────────────
    success: bool = Field(
        default=False,
        description="True if episode resolved AND closure reached (or mood-based success)",
    )
    repetition_count: int = Field(
        default=0,
        description="Consecutive repeated responses detected this episode",
    )
    stall_count: int = Field(
        default=0,
        description="Steps spent in the current stage without advancing",
    )
    failure_reason: str = Field(
        default="",
        description="Human-readable reason for episode failure (if done and not success)",
    )

    # ── Reward components ──────────────────────────────────────────────────
    rule_score: float = Field(
        default=0.0,
        description="Rule-based step quality score normalized to [0,1] — used internally by grader",
    )
    grader_score: float = Field(
        default=0.0,
        description="Calibrated episode score [0.0–1.0], populated only at done=True. reward==grader_score at episode end.",
    )
    reward_reason: str = Field(
        default="",
        description="Human-readable explanation of the reward (e.g. 'Resolved with closure')",
    )

    # ── Base fields for framework compatibility ───────────────────────────
    done: bool = Field(
        default=False,
        description="Whether the episode is finished",
    )
    reward: float = Field(
        default=0.0,
        description="The latest step's reward",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary per-step metadata",
    )

    model_config = {"extra": "allow"}
