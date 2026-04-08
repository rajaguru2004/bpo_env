"""
StagePolicyEnforcer — Per-stage required intent rule controller (internal only).

Augments the LLM system prompt with mandatory action instructions for each
conversation stage. Operates purely on the inference/agent side.
Does NOT modify environment state, rewards, or API responses.

Usage:
    hint = StagePolicyEnforcer.get_stage_hint(task_name, stage_name)
    if hint:
        system_prompt += f"\n\n[STAGE POLICY]: {hint}"
"""

from __future__ import annotations
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Policy definitions: task → stage → instruction
# ---------------------------------------------------------------------------

_STAGE_POLICIES: Dict[str, Dict[str, str]] = {
    # ── ORDER STATUS ───────────────────────────────────────────────────────
    "order_status": {
        "start": (
            "BEGIN with a warm acknowledgment or greeting. "
            "Show empathy that the customer is waiting for an update."
        ),
        "inquiry": (
            "MUST provide the tracking number (e.g., TRK987654321) AND the current "
            "order status (e.g., 'shipped', 'in transit'). Do not leave out either piece."
        ),
        "resolution": (
            "MUST include the expected delivery date AND confirm the tracking number. "
            "Be specific with dates (e.g., 'April 3rd' or 'within 2-3 business days')."
        ),
        "closure": (
            "MUST close the conversation professionally. Include a case/reference number, "
            "a confirmation phrase (e.g., 'Is there anything else I can help you with?'), "
            "and a polite sign-off."
        ),
    },

    # ── DAMAGED PRODUCT ────────────────────────────────────────────────────
    "damaged_product": {
        "start": (
            "MUST open with a sincere apology. The customer received a damaged item — "
            "start with 'I sincerely apologize' or equivalent before anything else."
        ),
        "empathy": (
            "MUST express deep empathy and understanding of the customer's frustration. "
            "Use phrases like 'I completely understand how frustrating this must be.' "
            "Do not jump directly to solutions here."
        ),
        "diagnosis": (
            "MUST ask for or reference the order details (order number, description of damage). "
            "Include a specific question such as 'Could you confirm your order number?'"
        ),
        "resolution": (
            "MUST explicitly offer a replacement or refund AND provide a timeline "
            "(e.g., 'within 3-5 business days'). Include both the action and the timeframe."
        ),
        "closure": (
            "MUST provide a case/reference number and close politely. "
            "Confirm the action taken and thank the customer for their patience."
        ),
    },

    # ── ESCALATION ─────────────────────────────────────────────────────────
    "escalation": {
        "start": (
            "MUST immediately apologize sincerely and de-escalate the situation. "
            "The customer is very angry — acknowledge their frustration before anything else. "
            "Use 'I sincerely apologize' and 'I completely understand your frustration.'"
        ),
        "de_escalation": (
            "MUST demonstrate empathy and ownership. Say something like 'I take full "
            "responsibility for this situation' and 'I will personally ensure this is resolved.' "
            "Do not give generic apologies."
        ),
        "acknowledgement": (
            "MUST escalate explicitly: mention a manager, supervisor, or senior team member. "
            "Example: 'I am escalating this to our senior customer service manager right now.'"
        ),
        "resolution": (
            "MUST confirm the refund or resolution AND provide a timeline. "
            "Example: 'I will process your full refund within 3-5 business days.' "
            "Also mention any escalation that has been done."
        ),
        "closure": (
            "MUST provide a case/reference number, confirm what was done, and "
            "close warmly. Thank the customer for their patience with this difficult situation."
        ),
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class StagePolicyEnforcer:
    """
    Rule-based stage policy enforcer.

    Each stage defines WHAT the agent MUST include. If the stage policy
    is not followed, the environment will assign low reward.
    This class informs the agent before it generates its response.
    """

    @staticmethod
    def get_stage_hint(task_name: str, stage_name: str) -> Optional[str]:
        """
        Return the stage-specific mandatory-action hint.
        Returns None if no policy is defined for this task/stage.
        """
        return _STAGE_POLICIES.get(task_name, {}).get(stage_name)

    @staticmethod
    def build_policy_prompt(task_name: str, stage_name: str) -> str:
        """
        Build a complete policy injection string ready to append to system prompt.
        Returns empty string if no policy found.
        """
        hint = StagePolicyEnforcer.get_stage_hint(task_name, stage_name)
        if not hint:
            return ""
        return (
            f"\n\n[MANDATORY STAGE POLICY for '{stage_name.upper()}' stage]: {hint}\n"
            "You MUST follow this policy in your response. Failure to do so will result "
            "in a low score."
        )

    @staticmethod
    def get_all_policies() -> Dict[str, Dict[str, str]]:
        """Return all policies (for debugging/testing only)."""
        return _STAGE_POLICIES
