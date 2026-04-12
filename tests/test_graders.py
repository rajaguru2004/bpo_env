"""
tests/test_graders.py — Unit tests for BPO grader functions.

Covers:
  - Positive cases: proper responses score above threshold
  - Negative cases: false-positive keyword check, blank responses, off-topic
  - Boundary cases: score clamping, escalation single-turn cap
  - Score range invariant: all outputs strictly in (0.0, 1.0)

Run with:
    pytest tests/test_graders.py -v
"""

import sys
import os

# Ensure project root is on path so 'tasks' is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tasks import grade_order_status, grade_damaged_product, grade_escalation, grade_episode


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _in_range(score: float) -> bool:
    """Score must be strictly between 0 and 1 (validator requirement)."""
    return 0.0 < score < 1.0


# ─────────────────────────────────────────────────────────────────────────────
# grade_order_status
# ─────────────────────────────────────────────────────────────────────────────

class TestGradeOrderStatus:

    def test_perfect_response_high_score(self):
        """Full tracking + status + delivery date + greeting → ≥ 0.70"""
        resp = (
            "Hello! Your order has been shipped. Your tracking number is TRK987654321. "
            "The estimated delivery date is April 5th. Is there anything else I can help you with?"
        )
        score = grade_order_status(resp)
        assert score >= 0.70, f"Expected ≥0.70, got {score}"
        assert _in_range(score)

    def test_partial_response_medium_score(self):
        """Only tracking number → around 0.30"""
        resp = "Your tracking number is TRK123456. Please check it on our website."
        score = grade_order_status(resp)
        assert 0.20 <= score <= 0.50, f"Expected 0.20–0.50, got {score}"

    def test_blank_response_min_score(self):
        """Empty string → 0.01"""
        assert grade_order_status("") == 0.01

    def test_very_short_response_min_score(self):
        """Less than 4 words → 0.01"""
        assert grade_order_status("OK") == 0.01

    # ── False-positive regression tests ──────────────────────────────────────

    def test_no_false_positive_auxiliary_may(self):
        """'may' as auxiliary verb should NOT earn delivery date credit."""
        resp = "I may be able to help you with that request today."
        score = grade_order_status(resp)
        # No tracking, no status, no real date phrase, no greeting keyword, no case ref
        assert score < 0.20, f"'may' auxiliary triggered false positive: {score}"

    def test_no_false_positive_days_alone(self):
        """Bare 'days' outside date context should NOT earn delivery date credit."""
        resp = "It has been many days since your order was placed. Thank you for reaching out."
        score = grade_order_status(resp)
        # Should only get greeting credit (+0.15), not delivery date too
        assert score < 0.30, f"standalone 'days' triggered false positive: {score}"

    def test_no_false_positive_ref_substring(self):
        """'preferred' contains 'ref' substring — must NOT earn case-reference credit.
        Response: status + greeting (no tracking, no date, no case ref) → ≤ 0.40"""
        resp = "Your preferred option has been shipped. Order status: in transit."
        score = grade_order_status(resp)
        # Should get status (+0.25) but NOT case ref (+0.10)
        # "preferred" must not match "ref #" or "ref no"
        # Score: shipped/in transit = +0.25 → max 0.25 (no ref keyword in new list)
        assert score <= 0.40, f"'ref' substring triggered false positive: {score}"

    def test_score_always_in_range(self):
        """Score must always be strictly (0, 1) regardless of input."""
        cases = ["", "ok", "hello", "a" * 500, "tracking number TRK999 delivered by April"]
        for c in cases:
            s = grade_order_status(c)
            assert _in_range(s), f"Out-of-range score {s} for input: {c[:50]!r}"


# ─────────────────────────────────────────────────────────────────────────────
# grade_damaged_product
# ─────────────────────────────────────────────────────────────────────────────

class TestGradeDamagedProduct:

    def test_perfect_response_high_score(self):
        """Full apology + empathy + replacement + timeline + case ref → ≥ 0.70"""
        resp = (
            "I sincerely apologize for the damaged product. I completely understand how frustrating "
            "this inconvenience must be. I will arrange a replacement to be shipped within 3-5 days. "
            "Your reference number is REF #78901."
        )
        score = grade_damaged_product(resp)
        assert score >= 0.70, f"Expected ≥0.70, got {score}"
        assert _in_range(score)

    def test_apology_only_low_score(self):
        """Just saying sorry → around 0.25"""
        resp = "I'm sorry to hear about your damaged product."
        score = grade_damaged_product(resp)
        assert score <= 0.30, f"Expected ≤0.30, got {score}"

    def test_blank_response_min_score(self):
        assert grade_damaged_product("") == 0.01

    # ── False-positive regression tests ──────────────────────────────────────

    def test_no_false_positive_ref_substring(self):
        """'referral' must NOT earn case ref credit (requires 'reference number', 'ref #', etc.).
        However 'inconvenience' correctly matches empathy → apology+empathy = 0.45 is fine."""
        resp = "I apologize for the inconvenience. The referral process takes time."
        score = grade_damaged_product(resp)
        # Should get apology (+0.25) + empathy (+0.20) = 0.45
        # Must NOT get case ref (+0.15) since 'referral' ≠ 'reference number' / 'ref #'
        # Validated: 'referral' does not appear in the new case-ref keyword list
        assert score <= 0.50, f"'referral' score unexpectedly high: {score}"
        # Extra: must be strictly less than if case-ref had triggered (0.45 + 0.15 = 0.60)
        assert score < 0.60, f"Case-ref credit must not have applied: {score}"

    def test_no_false_positive_week_context(self):
        """'last week' or 'next week' (non-timeline) should NOT earn timeline credit."""
        resp = "I apologize. The product you received last week was damaged. I understand your frustration."
        score = grade_damaged_product(resp)
        # Should get apology (0.25) + empathy (0.20) = 0.45 max
        # 'last week' should NOT match the timeline credit
        assert score <= 0.50, f"'last week' triggered timeline false positive: {score}"

    def test_timeline_within_24_hours_scores(self):
        """'within 24 hours' is a specific timeline phrase and should score."""
        resp = (
            "I apologize for the inconvenience. I understand your frustration. "
            "Your replacement will arrive within 24 hours."
        )
        score = grade_damaged_product(resp)
        # apology + empathy + timeline = 0.25 + 0.20 + 0.15 = 0.60
        assert score >= 0.55, f"Specific timeline phrase missed: {score}"

    def test_score_always_in_range(self):
        cases = ["", "sorry", "I understand and we will replace it within 48 hours", "a" * 500]
        for c in cases:
            s = grade_damaged_product(c)
            assert _in_range(s), f"Out-of-range score {s} for: {c[:50]!r}"


# ─────────────────────────────────────────────────────────────────────────────
# grade_escalation
# ─────────────────────────────────────────────────────────────────────────────

class TestGradeEscalation:

    def test_single_turn_capped_at_60(self):
        """Step 1: Even a perfect response must be capped at 0.60."""
        resp = (
            "I sincerely apologize and I hear you. I will process your full refund immediately. "
            "I will connect you with our senior manager. Within 24 hours you will receive a "
            "confirmation. Your reference number is REF #12345. I assure you we will resolve this."
        )
        score = grade_escalation(resp, state={"step": 1})
        assert score <= 0.60, f"Single-turn cap failed: got {score}"
        assert _in_range(score)

    def test_multi_turn_can_exceed_60(self):
        """Step 2+: Same content should score above 0.60 without hard cap."""
        resp = (
            "I sincerely apologize and I hear you. I will process your full refund immediately. "
            "I will connect you with our senior manager. Within 24 hours you will receive a "
            "confirmation. Your reference number is REF #12345. I assure you we will resolve this."
        )
        score = grade_escalation(resp, state={"step": 2})
        assert score > 0.60, f"Multi-turn should exceed single-turn cap: got {score}"

    def test_apology_only_step1_low_score(self):
        """No refund, no manager, no timeline at step 1 → low score."""
        resp = "I sincerely apologize for the trouble you've experienced. I hear you."
        score = grade_escalation(resp, state={"step": 1})
        assert score <= 0.30, f"Expected low step-1 score: {score}"

    def test_blank_response_min_score(self):
        assert grade_escalation("") == 0.01

    def test_partial_manager_at_step1(self):
        """Jumping straight to manager at step 1 gets only 0.10 (not 0.20)."""
        resp = "I will connect you with our manager immediately. I apologize."
        score_step1 = grade_escalation(resp, state={"step": 1})
        score_step2 = grade_escalation(resp, state={"step": 2})
        # step 1 gets 0.10 for manager, step 2 gets 0.20 — step2 must be higher
        assert score_step2 > score_step1, "Step-2 manager credit should exceed step-1"

    def test_no_false_positive_ref_substring(self):
        """'ref' as substring must not earn case reference credit."""
        resp = "I apologize. I will reimburse your full refund within 24 hours. Rest assured."
        score = grade_escalation(resp, state={"step": 2})
        # Gets: apology(0.20) + refund(0.20) + timeline(0.15) + de-escalation(0.10) = 0.65
        # Must NOT get case ref (0.15) since no 'ref #' or 'reference number'
        assert score <= 0.70, f"Expected ≤0.70 without explicit case ref: {score}"

    def test_score_always_in_range(self):
        cases = [
            ("", {}),
            ("sorry", {"step": 1}),
            ("I apologize and will give a full refund and connect you with a manager", {"step": 3}),
        ]
        for resp, state in cases:
            s = grade_escalation(resp, state)
            assert _in_range(s), f"Out-of-range: {s} for {resp[:40]!r}"


# ─────────────────────────────────────────────────────────────────────────────
# grade_episode (top-level)
# ─────────────────────────────────────────────────────────────────────────────

class TestGradeEpisode:

    def _resolved_episode(self, task_name: str = "order_status"):
        """Helper: minimal resolved trajectory dict for testing."""
        return {
            "resolved": True,
            "closure_reached": True,
            "steps_taken": 3,
            "max_steps": 5,
            "final_stage": "closure",
            "final_mood": "satisfied",
            "trajectory": [
                {"completeness_score": 0.8, "detected_intents": {}},
                {"completeness_score": 0.9, "detected_intents": {}},
                {"completeness_score": 0.85, "detected_intents": {}},
            ],
        }

    def test_resolved_with_closure_high_score(self):
        """Fully resolved episode with closure should score ≥ 0.55."""
        score = grade_episode("order_status", self._resolved_episode())
        assert score >= 0.55, f"Expected ≥0.55, got {score}"
        assert _in_range(score)

    def test_empty_trajectory_min_score(self):
        """Empty list trajectory → 0.01."""
        assert grade_episode("order_status", []) == 0.01

    def test_unresolved_partial_score(self):
        """Unresolved episode with some stage progress → small positive score."""
        traj = {"resolved": False, "closure_reached": False, "steps_taken": 2,
                "max_steps": 5, "final_stage": "resolution", "final_mood": "neutral",
                "trajectory": []}
        score = grade_episode("order_status", traj)
        assert 0.01 <= score <= 0.20, f"Expected low partial score, got {score}"

    def test_max_steps_resolved_gets_efficiency_floor(self):
        """Hard task using all max_steps but resolved should still get ≥ 0.05 efficiency."""
        # steps_taken == max_steps → ratio=1.0 → raw_eff=0; floor should save it
        traj = {"resolved": True, "closure_reached": True, "steps_taken": 12,
                "max_steps": 12, "final_stage": "closure", "final_mood": "neutral",
                "trajectory": [{"completeness_score": 0.6} for _ in range(12)]}
        score = grade_episode("escalation", traj)
        # At minimum: resolution(0.45) + efficiency_floor(0.05) + mood(0.05) = 0.55
        assert score >= 0.50, f"Full-steps resolved should score ≥0.50, got {score}"

    def test_score_always_in_range_various_tasks(self):
        for task in ["order_status", "damaged_product", "escalation",
                     "task_easy", "task_medium", "task_hard"]:
            s = grade_episode(task, self._resolved_episode(task))
            assert _in_range(s), f"Out-of-range {s} for task {task}"
