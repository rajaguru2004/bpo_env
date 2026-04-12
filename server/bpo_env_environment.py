"""
Customer Support Environment Implementation — v6 (Completeness + Sequencing Reward).

Simulates real-world customer support conversations with:
  - Multi-intent classification: a single response can carry multiple intents
  - Flexible stage transitions: any matching intent advances the stage
  - Stage-skip: strong resolution-oriented responses can skip minor stages
  - Completeness validation: required fields per stage must be present in response
  - Sequence validation: information must be provided in the correct order
  - Tri-partite reward: 0.4*intent + 0.4*completeness + 0.2*sequence
  - Penalty rules: missing info, wrong sequence, vague/short responses
  - Strict but fair: partial answers are penalized, complete answers rewarded
  - Pure rule-based grader (no LLM dependency)
  - Normalized reward: always in [0.0, 1.0]
  - reward == grader_score at episode end (perfect alignment)
  - reward_reason explains every score with specific missing-field detail

Agent (LLM) acts as a customer support executive.
Three difficulty levels: easy / medium / hard.
"""

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

# ---------------------------------------------------------------------------
# Environment variable loading
# ---------------------------------------------------------------------------


env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
if os.path.exists(env_path):
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        with open(env_path) as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    name, value = line.split("=", 1)
                    os.environ[name.strip()] = value.strip().strip('"').strip("'")

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

try:
    from models import CustomerSupportAction, CustomerSupportObservation
    from server.grader import grade_episode
    from server.intents import extract_intents, get_bridge_intents
    from server.reward_shaper import shape_reward
    try:
        from server.stage_sequence_guard import StageSequenceGuard
    except ImportError:
        StageSequenceGuard = None
except ImportError:
    from ..models import CustomerSupportAction, CustomerSupportObservation
    from .grader import grade_episode
    from .intents import extract_intents, get_bridge_intents
    from .reward_shaper import shape_reward
    try:
        from .stage_sequence_guard import StageSequenceGuard
    except ImportError:
        StageSequenceGuard = None


# ===========================================================================
# MULTI-INTENT CLASSIFICATION (v4 — returns ALL matching intents)
# ===========================================================================

# Short-trigger keywords allow broad matching even in natural LLM prose.
# Each entry is (intent_label, list_of_trigger_phrases).
# Phrases are matched case-insensitively as substrings.
_INTENT_KEYWORDS: Dict[str, List[str]] = {
    # ── Greeting / Acknowledgment ─────────────────────────────────────────
    "greeting": [
        "hello", "hi there", "good morning", "good afternoon", "good evening",
        "welcome", "how can i help", "how may i help", "happy to assist",
        "glad to help", "thank you for reaching out", "thank you for contacting",
        "thank you for calling", "i'd be happy to help", "i will be happy to help",
        "i'm here to help", "i am here to help",
    ],
    # ── Apology ───────────────────────────────────────────────────────────
    "apology": [
        "i sincerely apologize", "i deeply apologize", "i am truly sorry",
        "i'm truly sorry", "i'm so sorry", "i am so sorry",
        "i apologize", "my sincerest apologies", "please accept my apologies",
        "i'm sorry to hear", "i am sorry to hear", "i regret",
        "we sincerely apologize", "we are sorry", "we're sorry",
        "deeply sorry", "sincerely sorry", "sorry for", "apologize for",
        "sorry about", "i'm sorry",
    ],
    # ── Empathy / De-escalation ───────────────────────────────────────────
    "de_escalation": [
        "i completely understand", "i totally understand", "i understand your frustration",
        "i understand how frustrating", "i hear you", "i can see why",
        "you're absolutely right", "you are absolutely right",
        "i take full responsibility", "i take ownership",
        "i assure you", "rest assured", "i promise",
        "totally understandable", "valid concern", "i will personally",
        "let me personally", "i will make sure", "i will ensure",
        "your concern is valid", "i acknowledge", "understandable",
        "i understand", "i can imagine", "i can see", "that must be",
        "i appreciate your patience", "i value your",
    ],
    # ── Information request / Clarification ──────────────────────────────
    "information_request": [
        "could you please provide", "could you provide", "could you share",
        "can you provide", "can you share", "may i have",
        "may i ask for", "please share", "please provide",
        "what is your order", "what is the order number",
        "order number", "could you confirm your",
        "can you confirm", "what was the", "could you tell me",
        "could you clarify", "can you clarify",
        "please confirm", "could you describe", "can you describe",
        "could you", "can you give me", "would you mind",
    ],
    # ── Resolution / Solution offer ───────────────────────────────────────
    "resolution_offer": [
        "i will arrange", "i will process", "i will initiate",
        "i will send", "i will ship", "i will dispatch",
        "i'll arrange", "i'll process", "i'll initiate",
        "i'll send", "i'll ship", "i'll dispatch",
        "we will arrange", "we will process", "we will send",
        "replacement", "full refund", "issue a refund",
        "process your refund", "send a new", "new unit",
        "arrange a replacement", "arrange a refund",
        "ship out a new", "re-send", "dispatch a",
        "credit your account", "compensation",
        "escalate this to", "connect you with a supervisor",
        "transfer you to", "pass this to our manager",
        "speak with a manager", "supervisor will",
        "i will resolve", "i'll resolve", "let me resolve",
        "i will fix", "i'll fix", "expedite",
        "priority shipping", "immediate action",
        "refund", "replace", "exchange",
    ],
    # ── Information provide (order/tracking details) ──────────────────────
    "information_provide": [
        "tracking number", "tracking id", "trk", "shipment tracking",
        "order status", "your order has been shipped", "order is shipped",
        "order has shipped", "on its way", "in transit",
        "expected delivery", "estimated delivery", "estimated arrival",
        "delivery date", "dispatch date", "shipped on",
        "order #", "order number is", "your order",
        "status of your order", "current status",
    ],
    # ── Confirmation / Closure ─────────────────────────────────────────────
    "confirmation": [
        "your case number", "your reference number", "reference number",
        "case number", "ticket number", "case id",
        "has been confirmed", "has been processed", "has been initiated",
        "is confirmed", "is processed", "is completed",
        "you will receive a confirmation", "you will receive an email",
        "within 24 hours", "within 48 hours", "within 3-5 business days",
        "within 2-3 business days", "business days",
        "is there anything else", "anything else i can help",
        "happy to help further", "glad i could assist",
        "thank you for your patience", "thank you for contacting",
        "have a wonderful day", "have a great day",
        "is resolved", "has been resolved", "all set",
        "you're all set", "you are all set", "take care",
    ],
}


def classify_intents(response: str) -> List[str]:
    """
    Classify all intents present in the agent response using keyword matching.

    Returns a list of matching intent labels (may be empty → ['off_topic']).
    Order reflects priority (confirmation first, greeting last) but ALL matches
    are returned, enabling multi-intent stage transitions.
    """
    lower = response.lower()

    # Priority order for deduplication in display, but we return ALL matches
    priority_order = [
        "confirmation",
        "resolution_offer",
        "information_provide",
        "information_request",
        "de_escalation",
        "apology",
        "greeting",
    ]

    matched: List[str] = []
    for intent in priority_order:
        keywords = _INTENT_KEYWORDS.get(intent, [])
        if any(kw in lower for kw in keywords):
            matched.append(intent)

    return matched if matched else ["off_topic"]


def classify_intent(response: str) -> str:
    """
    Backwards-compatible single-label intent classifier.
    Returns the highest-priority intent (first element of classify_intents).
    """
    return classify_intents(response)[0]


# ===========================================================================
# REPETITION DETECTION (relaxed — Jaccard threshold raised to 0.85)
# ===========================================================================

def _response_fingerprint(text: str) -> str:
    """Normalize and fingerprint a response for repetition detection."""
    normalized = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.md5(normalized.encode()).hexdigest()


def _jaccard_similarity(a: str, b: str) -> float:
    """Compute word-level Jaccard similarity between two strings."""
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _is_repetitive(
    response: str,
    intents: List[str],
    prev_intents: List[List[str]],
    prev_responses: List[str],
    prev_fingerprints: List[str],
) -> Tuple[bool, str]:
    """
    Return (is_repetitive, reason).

    Detects:
    - Exact fingerprint match (same normalized text)
    - Near-identical response (Jaccard similarity > 0.85) in last 2 responses
    - Same intents AND near-identical text (combined check)

    NOTE: Same intent alone is no longer flagged as repetitive.
    The agent may need to apologize twice but with different content.
    """
    current_fp = _response_fingerprint(response)

    # Exact fingerprint match against last 3 responses
    if current_fp in prev_fingerprints[-3:]:
        return True, "exact_repeat"

    # Near-identical text (Jaccard > 0.85) against last 2 responses
    for prev_resp in prev_responses[-2:]:
        if _jaccard_similarity(response, prev_resp) > 0.85:
            return True, "near_identical_text"

    return False, ""


# ===========================================================================
# STAGE DEFINITIONS (relaxed — broader accepted_intents, higher max_stall)
# ===========================================================================

TASK_STAGES: Dict[str, List[Dict[str, Any]]] = {
    "order_status": [
        {
            "name": "start",
            "accepted_intents": {
                "greeting", "apology", "information_provide",
                "information_request", "de_escalation",
            },
            "advance_intents": {
                "greeting", "apology", "information_provide", "information_request",
            },
            "hint": "Greet the customer and acknowledge their order inquiry.",
            "max_stall": 2,
        },
        {
            "name": "inquiry",
            "accepted_intents": {"information_provide", "information_request", "confirmation"},
            "advance_intents": {"information_provide"},
            "hint": "Provide the tracking number and current order status.",
            "max_stall": 2,
        },
        {
            "name": "resolution",
            "accepted_intents": {"confirmation", "information_provide", "resolution_offer"},
            "advance_intents": {"confirmation", "information_provide"},
            "hint": "Confirm the expected delivery date and next steps.",
            "max_stall": 2,
        },
        {
            "name": "closure",
            "accepted_intents": {"confirmation", "de_escalation", "greeting"},
            "advance_intents": {"confirmation"},
            "hint": "Close the conversation professionally.",
            "max_stall": 2,
        },
    ],
    "damaged_product": [
        {
            "name": "start",
            # Broader: greeting + apology is realistic; de_escalation also accepted
            "accepted_intents": {"apology", "greeting", "de_escalation", "information_request"},
            "advance_intents": {"apology", "de_escalation"},
            "hint": None,
            "max_stall": 2,
        },
        {
            "name": "empathy",
            "accepted_intents": {
                "apology", "de_escalation", "resolution_offer", "information_request",
            },
            "advance_intents": {"de_escalation", "resolution_offer", "information_request"},
            "hint": None,
            "max_stall": 3,
        },
        {
            "name": "diagnosis",
            "accepted_intents": {
                "information_request", "de_escalation", "resolution_offer",
            },
            "advance_intents": {"information_request", "resolution_offer"},
            "hint": None,
            "max_stall": 3,
        },
        {
            "name": "resolution",
            "accepted_intents": {"resolution_offer", "confirmation", "information_provide"},
            "advance_intents": {"resolution_offer", "confirmation"},
            "hint": None,
            "max_stall": 3,
        },
        {
            "name": "closure",
            "accepted_intents": {"confirmation", "de_escalation", "greeting"},
            "advance_intents": {"confirmation"},
            "hint": None,
            "max_stall": 2,
        },
    ],
    "escalation": [
        {
            "name": "start",
            "accepted_intents": {
                "apology", "de_escalation", "greeting", "resolution_offer",
            },
            "advance_intents": {"apology", "de_escalation", "resolution_offer"},
            "hint": None,
            "max_stall": 3,
        },
        {
            "name": "de_escalation",
            "accepted_intents": {
                "de_escalation", "apology", "resolution_offer", "information_request",
            },
            "advance_intents": {"de_escalation", "resolution_offer"},
            "hint": None,
            "max_stall": 3,
        },
        {
            "name": "acknowledgement",
            "accepted_intents": {
                "resolution_offer", "de_escalation", "information_provide",
            },
            "advance_intents": {"resolution_offer"},
            "hint": None,
            "max_stall": 3,
        },
        {
            "name": "resolution",
            "accepted_intents": {
                "resolution_offer", "confirmation", "information_provide",
            },
            "advance_intents": {"resolution_offer", "confirmation"},
            "hint": None,
            "max_stall": 3,
        },
        {
            "name": "closure",
            "accepted_intents": {"confirmation", "de_escalation", "greeting"},
            "advance_intents": {"confirmation"},
            "hint": None,
            "max_stall": 3,
        },
    ],
}

# Stage-skip map: if any of these intents appear, jump directly to target_stage
# regardless of current stage (only if current stage index < target_stage index)
STAGE_SKIP_RULES: List[Dict[str, Any]] = [
    {
        "trigger_intents": {"resolution_offer"},
        "target_stage": "resolution",
        "tasks": {"damaged_product", "escalation"},
        "min_stage_idx": 1,  # must have progressed past start
    },
    {
        "trigger_intents": {"confirmation"},
        "target_stage": "closure",
        "tasks": {"order_status", "damaged_product", "escalation"},
        "min_stage_idx": 2,  # must be at least past diagnosis/empathy
    },
]


# ===========================================================================
# REQUIRED FIELDS REGISTRY (per task, per stage)
# Defines what information MUST appear in a response to be considered complete.
# Each field entry is a list of keyword alternatives (OR logic).
# All entries in the list must be present (AND logic between entries).
# ===========================================================================

# Structure:
#   REQUIRED_FIELDS[task_name][stage_name] = [
#       [kw_alt1, kw_alt2, ...],   # field 1 — any one keyword satisfies it
#       [kw_alt1, kw_alt2, ...],   # field 2 — any one keyword satisfies it
#       ...                         # ALL fields must be satisfied for score=1.0
#   ]

REQUIRED_FIELDS: Dict[str, Dict[str, List[List[str]]]] = {
    "order_status": {
        # start: just greeting/ack — no hard content requirement
        "start": [],
        # inquiry: must provide tracking number AND/OR current order status
        "inquiry": [
            ["tracking number", "tracking id", "trk", "tracking #", "shipment tracking"],  # field A: tracking
            ["order status", "shipped", "in transit", "on its way",
             "out for delivery", "dispatched", "processed"],                                 # field B: status
        ],
        # resolution: must provide tracking AND delivery date
        "resolution": [
            ["tracking number", "tracking id", "trk", "tracking #"],                        # field A: tracking
            ["expected delivery", "estimated delivery", "delivery date",
             "arrive", "arrival", "deliver by", "delivered by",
             "april", "march", "may", "days"],                                               # field B: delivery date
        ],
        # closure: must include confirmation/sign-off
        "closure": [
            ["case number", "reference number", "ticket number",
             "anything else", "have a", "take care", "resolved",
             "thank you", "my pleasure", "you're welcome"],                                  # field A: closure phrase
        ],
    },
    "damaged_product": {
        "start": [],
        # empathy: must apologize and show empathy
        "empathy": [
            ["apologize", "sorry", "apologies", "regret"],                                  # field A: apology word
            ["understand", "hear you", "frustrat", "inconvenience",
             "concern", "appreciate your patience"],                                         # field B: empathy phrase
        ],
        # diagnosis: must ask for or reference order/product details
        "diagnosis": [
            ["order number", "order #", "order id", "98765",
             "photo", "describe", "details", "can you confirm",
             "what happened", "damaged", "broken"],                                          # field A: order/damage ref
        ],
        # resolution: must offer replacement or refund with timeline
        "resolution": [
            ["replacement", "replace", "new unit", "refund",
             "send a new", "ship a new", "arrange"],                                         # field A: action offered
            ["days", "hours", "week", "business days",
             "24", "48", "3-5", "timeline", "when"],                                         # field B: timeline
        ],
        # closure: must give case/reference number and sign off
        "closure": [
            ["case number", "reference number", "ticket number",
             "case id", "ref", "number is", "confirmation",
             "anything else", "have a", "take care", "thank you"],                           # field A: ref or sign-off
        ],
    },
    "escalation": {
        "start": [],
        # de_escalation: must sincerely apologize and show empathy
        "de_escalation": [
            ["apologize", "sorry", "apologies", "sincerely"],                               # field A: apology
            ["understand", "frustrat", "hear you", "concern",
             "appreciate", "regret", "i acknowledge"],                                       # field B: empathy
        ],
        # acknowledgement: must escalate or mention manager/supervisor
        "acknowledgement": [
            ["escalate", "supervisor", "manager", "senior",
             "team lead", "specialist", "pass this",
             "transfer", "connect you with"],                                                # field A: escalation action
        ],
        # resolution: must offer refund/resolution with a timeline
        "resolution": [
            ["refund", "full refund", "replace", "compensation",
             "credit", "resolve", "process"],                                                # field A: resolution type
            ["days", "hours", "24", "48", "3-5", "week",
             "business days", "timeline", "within"],                                         # field B: timeline
        ],
        # closure: give case number and close
        "closure": [
            ["case number", "reference number", "ticket number",
             "case id", "confirmation", "ref",
             "anything else", "have a", "take care", "thank you"],                           # field A: closure
        ],
    },
}


# ===========================================================================
# SEQUENCE VIOLATION RULES
# Detect when a response provides information that is only correct AFTER
# a certain stage has been reached.
# ===========================================================================

# Each rule: if stage_index < required_stage_idx AND any trigger keyword found → penalty
SEQUENCE_RULES: List[Dict[str, Any]] = [
    {
        # Should not give status info without tracking info (order_status)
        "task": "order_status",
        "max_stage_idx": 1,
        "trigger_keywords": [
            "shipped", "in transit", "on its way", "dispatched", "processed", 
            "out for delivery", "status is", "order is",
        ],
        "prerequisite_keywords": [
            "tracking number", "tracking id", "trk", "tracking #",
        ],
        "penalty": 0.8,
        "reason": "Provided status info without providing a tracking number.",
    },
    {
        # Should not give delivery date at inquiry stage (must give tracking first)
        "task": "order_status",
        "max_stage_idx": 1,          # Only applies when at stage 0 or 1 (start/inquiry)
        "trigger_keywords": [
            "expected delivery", "delivery date", "estimated delivery",
            "deliver by", "arrive by", "arrival date", "arrive on",
            "delivery on", "estimated arrival", "arrival at",
            "arrive", "delivery",  # broader triggers
        ],
        "prerequisite_keywords": [
            "tracking number", "tracking id", "trk", "tracking #",
        ],
        "penalty": 0.4,              # sequence_score is reduced by this
        "reason": "Provided delivery date before confirming tracking number.",
    },
    {
        # Should not offer resolution before showing empathy (damaged_product)
        "task": "damaged_product",
        "max_stage_idx": 1,          # start or empathy stage
        "trigger_keywords": [
            "replacement", "refund", "send a new", "ship a new",
        ],
        "prerequisite_keywords": [
            "apologize", "sorry", "apologies", "understand",
        ],
        "penalty": 0.3,
        "reason": "Jumped to resolution before showing empathy.",
    },
    {
        # Should not close before resolution (all tasks)
        "task": None,                # applies to all tasks
        "max_stage_idx": 1,          # any early stage
        "trigger_keywords": [
            "case number", "reference number", "ticket number",
            "have a wonderful day", "have a great day", "all set",
        ],
        "prerequisite_keywords": [],  # no prerequisite — just flag the premature closure
        "penalty": 0.5,
        "reason": "Premature closure — jumped to end before resolving the issue.",
    },
]


# ===========================================================================
# COMPLETENESS VALIDATION
# ===========================================================================

def check_completeness(
    response: str,
    task_name: str,
    stage_name: str,
) -> Tuple[float, List[str]]:
    """
    Validate that the response contains all required fields for the given stage.

    Returns:
        completeness_score (float in [0.0, 1.0]): fraction of required fields satisfied.
        missing_fields (List[str]): human-readable descriptions of missing fields.
    """
    task_fields = REQUIRED_FIELDS.get(task_name, {})
    stage_fields: List[List[str]] = task_fields.get(stage_name, [])

    # No requirements defined → full credit (neutral stage)
    if not stage_fields:
        return 1.0, []

    lower = response.lower()
    satisfied = 0
    missing: List[str] = []

    for idx, keyword_group in enumerate(stage_fields):
        # Each group is a list of keyword alternatives (OR logic)
        if any(kw in lower for kw in keyword_group):
            satisfied += 1
        else:
            # Build a short human-readable label for the missing field
            # Use first keyword in group as representative
            rep = keyword_group[0] if keyword_group else f"field_{idx+1}"
            missing.append(rep)

    score = satisfied / len(stage_fields)
    return score, missing


# ===========================================================================
# SEQUENCE VALIDATION
# ===========================================================================

def check_sequence(
    response: str,
    task_name: str,
    stage_index: int,
) -> Tuple[float, str]:
    """
    Validate that the response follows the correct information order.

    Returns:
        sequence_score (float in [0.0, 1.0]): 1.0 = correct order, lower = violation.
        violation_reason (str): human-readable reason if violated, else empty string.
    """
    lower = response.lower()
    violations: List[str] = []
    total_penalty = 0.0

    for rule in SEQUENCE_RULES:
        # Check task match (None = all tasks)
        if rule["task"] is not None and rule["task"] != task_name:
            continue
        # Only apply if we are at or below the max_stage_idx threshold
        if stage_index > rule["max_stage_idx"]:
            continue
        # Check if trigger keywords are present
        trigger_hit = any(kw in lower for kw in rule["trigger_keywords"])
        if not trigger_hit:
            continue
        # Check if prerequisite keywords are present (which would make it acceptable)
        prereqs = rule.get("prerequisite_keywords", [])
        prereq_met = any(kw in lower for kw in prereqs) if prereqs else False
        if prereqs and prereq_met:
            # Prerequisites are present → this is actually correct ordering
            continue
        # Violation detected
        violations.append(rule["reason"])
        total_penalty = max(total_penalty, rule["penalty"])

    if violations:
        sequence_score = max(0.0, 1.0 - total_penalty)
        return sequence_score, " | ".join(violations)
    return 1.0, ""


def _get_skip_target(
    task_name: str,
    intents: Set[str],
    current_stage_idx: int,
    stages: List[Dict[str, Any]],
) -> Optional[int]:
    """
    Return the new stage index if a skip rule applies, else None.
    Only skips forward; never moves backward.
    """
    stage_names = [s["name"] for s in stages]
    for rule in STAGE_SKIP_RULES:
        if task_name not in rule["tasks"]:
            continue
        if current_stage_idx < rule["min_stage_idx"]:
            continue
        if rule["trigger_intents"] & intents:
            target_name = rule["target_stage"]
            if target_name in stage_names:
                target_idx = stage_names.index(target_name)
                if target_idx > current_stage_idx:
                    return target_idx
    return None


# ===========================================================================
# CUSTOMER SCRIPTS (deterministic, stage × mood)
# ===========================================================================

CUSTOMER_SCRIPTS: Dict[str, Dict[str, Dict[str, List[str]]]] = {
    "order_status": {
        "start": {
            "angry":     ["My order is LATE and nobody is helping me!"],
            "neutral":   ["Hi, I placed an order (#12345) three days ago and haven't "
                          "received any shipping confirmation yet. Can you check the status?"],
            "satisfied": ["Could you update me on my order #12345?"],
        },
        "inquiry": {
            "angry":     ["That's not good enough. Give me the tracking number NOW."],
            "neutral":   ["Can you tell me the tracking number too?"],
            "satisfied": ["Great, could you share the tracking number please?"],
        },
        "resolution": {
            "angry":     ["When exactly will it arrive? I need a specific date!"],
            "neutral":   ["When will it be delivered approximately?"],
            "satisfied": ["Wonderful, just wanted to confirm the delivery date."],
        },
        "closure": {
            "angry":     ["Fine. I hope this doesn't happen again."],
            "neutral":   ["Thank you, that's all I needed."],
            "satisfied": ["Perfect, thank you for the quick help!"],
        },
    },
    "damaged_product": {
        "start": {
            "angry":     ["I received my package today but the product inside was completely damaged. "
                          "The box was crushed and the item is broken. This is UNACCEPTABLE! "
                          "I need a replacement or a refund immediately."],
            "neutral":   ["I received my package today but the product inside was completely damaged. "
                          "The box was crushed and the item is broken. "
                          "I need a replacement or a refund."],
            "satisfied": ["Unfortunately my order arrived damaged. Can you help?"],
        },
        "empathy": {
            "angry":     ["I don't want to hear apologies, I want action! Send me a replacement NOW!"],
            "neutral":   ["I appreciate the apology. What do I need to do to get a replacement?"],
            "satisfied": ["Thank you for understanding. What's the process?"],
        },
        "diagnosis": {
            "angry":     ["Of course it's broken! The box was completely crushed when it arrived!"],
            "neutral":   ["The product was broken on arrival — the case was shattered."],
            "satisfied": ["Yes, it was damaged. Order #98765."],
        },
        "resolution": {
            "angry":     ["How long will the replacement take? I needed this urgently!"],
            "neutral":   ["How long will the replacement take to arrive?"],
            "satisfied": ["That's great. What's the expected delivery for the replacement?"],
        },
        "closure": {
            "angry":     ["Okay, but this better not happen again. Give me a reference number."],
            "neutral":   ["Alright, thank you for resolving this."],
            "satisfied": ["Excellent — thank you for handling this so well!"],
        },
    },
    "escalation": {
        "start": {
            "angry":     ["This is absolutely ridiculous! I've been waiting for 2 weeks and my "
                          "order STILL hasn't arrived. I've called 3 times already and nobody "
                          "helps me! I want a FULL REFUND and I want to speak to your manager RIGHT NOW!"],
            "neutral":   ["I've been waiting 2 weeks for my order. I'd like a refund or to escalate."],
            "satisfied": ["Could someone senior handle my overdue order situation?"],
        },
        "de_escalation": {
            "angry":     ["You're just reading from a script! I want REAL answers, not generic apologies!"],
            "neutral":   ["I understand you're sorry, but what are you actually going to do about it?"],
            "satisfied": ["I appreciate the empathy. What are the next steps?"],
        },
        "acknowledgement": {
            "angry":     ["Nobody has taken any responsibility for this mess. "
                          "Can you ACTUALLY help me or do I need to go to social media?"],
            "neutral":   ["Can I speak to a manager directly?"],
            "satisfied": ["Okay, what options do I have for resolution?"],
        },
        "resolution": {
            "angry":     ["How long will the refund take? I need it in writing!"],
            "neutral":   ["Fine, if you can confirm the refund and escalate, I'll wait."],
            "satisfied": ["That sounds reasonable — please proceed."],
        },
        "closure": {
            "angry":     ["I still think this was handled poorly, but okay."],
            "neutral":   ["Okay, I appreciate you handling this. Thank you."],
            "satisfied": ["Thank you for resolving this. I'm satisfied."],
        },
    },
}


# ===========================================================================
# TASK DEFINITIONS
# ===========================================================================

TASKS: Dict[str, Dict[str, Any]] = {
    "order_status": {
        "difficulty": "easy",
        "max_steps": 5,
        "description": "Customer wants to know the status of their order #12345.",
        "initial_mood": "neutral",
        "context": {
            "order_id": "12345",
            "order_date": "2026-03-29",
            "status": "Shipped",
            "tracking_number": "TRK987654321",
            "expected_delivery": "2026-04-03",
        },
        "provide_hints": True,
        "max_consecutive_failures": 4,    # relaxed from 3
        "max_consecutive_repetitions": 3, # relaxed from 2
    },
    "damaged_product": {
        "difficulty": "medium",
        "max_steps": 8,
        "description": "Customer received a damaged product and wants a replacement.",
        "initial_mood": "angry",
        "context": {
            "order_id": "98765",
            "product": "Bluetooth Speaker",
            "issue": "Damaged on arrival",
        },
        "provide_hints": False,
        "max_consecutive_failures": 4,
        "max_consecutive_repetitions": 3,
        "no_apology_by_step": 3,         # relaxed from 2 — must apologize within 3 steps
    },
    "escalation": {
        "difficulty": "hard",
        "max_steps": 12,
        "description": (
            "Angry customer demanding a full refund and to speak with a manager. "
            "Multi-turn de-escalation scenario."
        ),
        "initial_mood": "angry",
        "context": {
            "order_id": "55501",
            "product": "Laptop Stand",
            "days_delayed": 14,
            "prior_contacts": 3,
        },
        "provide_hints": False,
        "max_consecutive_failures": 4,
        "max_consecutive_repetitions": 3,
    },
}

TASK_REQUIREMENTS = {
    "order_status": {
        "required": ["tracking_info", "delivery_info"],
        "optional": ["empathy", "closure"]
    },
    "damaged_product": {
        "required": ["empathy", "replacement"],
        "optional": ["closure"]
    },
    "escalation": {
        "required": ["empathy", "escalation", "refund"],
        "optional": ["closure"]
    }
}

# ---------------------------------------------------------------------------
# STEP 6: Expected stage flow for progression reward
# ---------------------------------------------------------------------------
EXPECTED_FLOW: Dict[str, List[str]] = {
    "order_status":   ["start", "inquiry", "resolution", "closure"],
    "escalation":     ["start", "de_escalation", "acknowledgement", "resolution", "closure"],
    "damaged_product":["start", "empathy", "diagnosis", "resolution", "closure"],
}


# ===========================================================================
# LLM JUDGE (called ONCE per episode — contributes to grader hybrid score)
# ===========================================================================

def _llm_judge_score(
    task_description: str,
    conversation_history: List[Dict[str, str]],
    final_response: str,
) -> float:
    """
    Call LLM via OpenRouter to evaluate the entire episode quality.
    Called ONCE per episode (not per step). Returns normalized [0,1] score.
    Falls back to 0.0 on any error (grader then uses pure rule score).
    """
    try:
        from openai import OpenAI

        llm_base_url = os.getenv("LLM_BASEURL")
        api_key = os.getenv("OPENAI_API_KEY")
        model_name = os.getenv("MODEL_NAME")

        if not all([llm_base_url, api_key, model_name]):
            return 0.0

        client = OpenAI(base_url=llm_base_url, api_key=api_key)

        convo_str = "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in conversation_history[-6:]
        )

        judge_prompt = f"""You are an expert customer service quality evaluator.

Task: {task_description}

Conversation Summary (last 6 turns):
{convo_str}

Rate the agent's OVERALL performance (0–10 each):
1. Helpfulness: Did the agent resolve the customer's issue?
2. Empathy: Was the agent professional and empathetic throughout?
3. Efficiency: Did the agent resolve the issue without unnecessary delays?

Respond ONLY with valid JSON:
{{"helpfulness": <0-10>, "empathy": <0-10>, "efficiency": <0-10>}}"""

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a customer service quality evaluator. Respond with valid JSON only.",
                },
                {"role": "user", "content": judge_prompt},
            ],
            max_tokens=80,
            temperature=0.1,
        )

        content = response.choices[0].message.content.strip()
        json_match = re.search(r"\{[^}]+\}", content)
        if json_match:
            scores = json.loads(json_match.group())
            avg = (
                float(scores.get("helpfulness", 5))
                + float(scores.get("empathy", 5))
                + float(scores.get("efficiency", 5))
            ) / 3.0
            return min(1.0, max(0.0, avg / 10.0))

        return 0.0
    except Exception:
        return 0.0


# ===========================================================================
# EPISODE STATE
# ===========================================================================

@dataclass
class EpisodeState:
    task_name: str
    stages: List[Dict[str, Any]]            # full stage list from TASK_STAGES
    stage_index: int = 0                     # current position in stage list
    customer_mood: str = "neutral"
    issue_status: str = "unresolved"         # unresolved | in_progress | resolved
    steps_taken: int = 0
    max_steps: int = 10
    # Failure counters (relaxed limits)
    consecutive_failures: int = 0           # wrong-intent steps in a row
    consecutive_repetitions: int = 0        # repeated response steps
    low_reward_streak: int = 0              # steps in a row with < 0.3 reward (Task 5.C)
    stall_steps: int = 0                    # steps without advancing current stage
    # Resolution tracking
    resolved: bool = False
    had_apology: bool = False
    closure_reached: bool = False
    # Recovery tracking
    prev_step_was_wrong: bool = False       # enables recovery bonus next step
    failure_recovery_active: bool = False   # True when last step reward < 0.3 → force recovery
    # History for repetition detection
    intent_history: List[List[str]] = field(default_factory=list)  # list of intent lists
    response_history: List[str] = field(default_factory=list)       # raw response texts
    response_fingerprints: List[str] = field(default_factory=list)
    # Trajectory for grader
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    # Failure reason
    failure_reason: str = ""
    # Limits (loaded from task config)
    max_consecutive_failures: int = 4
    max_consecutive_repetitions: int = 3
    # STEP 2: Episode-level intent accumulator — tracks which required intents
    # have been confidently detected at least once during the episode.
    collected_intents: Dict[str, bool] = field(default_factory=lambda: {
        "tracking_info": False,
        "delivery_info": False,
        "empathy":        False,
        "escalation":     False,
        "refund":         False,
        "replacement":    False,
    })

    @property
    def current_stage(self) -> Dict[str, Any]:
        idx = min(self.stage_index, len(self.stages) - 1)
        return self.stages[idx]

    @property
    def stage_name(self) -> str:
        return self.current_stage["name"]

    @property
    def at_final_stage(self) -> bool:
        return self.stage_index >= len(self.stages) - 1

    @property
    def done(self) -> bool:
        return (
            self.resolved
            or self.consecutive_failures >= self.max_consecutive_failures
            or self.consecutive_repetitions >= self.max_consecutive_repetitions
            or self.steps_taken >= self.max_steps
        )


# ===========================================================================
# REWARD COMPUTATION
# ===========================================================================

# ===========================================================================
# STEP REWARD — v6 Tri-Partite Formula
# final_score = 0.4*intent_score + 0.4*completeness_score + 0.2*sequence_score
# Each component is in [0.0, 1.0]; final_score clamped to [0.0, 1.0].
#
# Intent score mapping:
#   All intents match accepted set (≥2 or sole match): 1.0
#   Partial match (1 of many):                         0.6
#   Wrong intent (no match):                           0.1
#   Off-topic only:                                    0.0
#   Repetitive:                                        0.0
#
# Completeness: derived from check_completeness() [0.0–1.0]
#   Short/vague response (<10 words) caps completeness at 0.3
#
# Sequence: derived from check_sequence() [0.0–1.0]
#   0.0 = severe out-of-order; 1.0 = correct ordering
#
# Stall penalty: -0.1 subtracted from final_score when stalling
# Repetition:    intent_score=0, completeness_score=0 → near-zero reward
# ===========================================================================

def _compute_step_reward(
    response_text: str,
    intents: List[str],
    intents_set: Set[str],
    accepted_intents: Set[str],
    stage_advanced: bool,
    is_repetitive: bool,
    is_stalling: bool,
    prev_step_was_wrong: bool,  # kept for API compat
    task_name: str = "",
    stage_name: str = "",
    stage_index: int = 0,
    stall_count: int = 0,
    detected_intents_dict: Dict[str, Any] = None,
    history_intents: Optional[List[Set[str]]] = None,
    last_reward: float = 1.0,
    low_reward_streak: int = 0,
) -> Tuple[float, float, float, float, str]:
    """
    Core Step Reward Function (v4 — Sharpened).
    Returns (step_reward, rule_score [0,1], completeness_score [0,1],
             sequence_score [0,1], reason_string).

    step_reward = final tri-partite score clamped to [0.0, 1.0].
    rule_score  = backward-compat alias for step_reward (same value).
    """
    words = response_text.split()
    reason_parts: List[str] = []

    # ── 1. INTENT SCORE ──────────────────────────────────────────────────────
    if is_repetitive:
        intent_score = 0.0
        reason_parts.append("Repetitive response.")
    elif "off_topic" in intents and len(intents) == 1:
        intent_score = 0.0
        reason_parts.append("Response is off-topic.")
    else:
        matching = intents_set & accepted_intents
        if len(matching) == 0:
            intent_score = 0.1
            reason_parts.append("Response doesn't match expected intent for this stage.")
        elif len(matching) >= 2 or (len(intents_set) == 1 and matching):
            intent_score = 1.0
        else:
            intent_score = 0.6
            reason_parts.append("Partial intent match.")

    # ── 2. COMPLETENESS SCORE ────────────────────────────────────────────────
    if is_repetitive:
        completeness_score = 0.0
        missing_fields: List[str] = []
    else:
        completeness_score, missing_fields = check_completeness(
            response_text, task_name, stage_name
        )
        # Vague / too-short response caps completeness at 0.3
        if len(words) < 10:
            completeness_score = min(completeness_score, 0.3)
            # Also cap intent_score heavily for short, low-effort responses
            intent_score = min(intent_score, 0.1)
            reason_parts.append("Response is too brief or vague.")
        
        # If missing critical info, also cap intent_score
        if completeness_score < 0.5:
            intent_score = min(intent_score, 0.4)

        if missing_fields:
            reason_parts.append(
                "Missing: " + ", ".join(missing_fields) + "."
            )

    # ── 3. SEQUENCE SCORE ────────────────────────────────────────────────────
    if is_repetitive:
        sequence_score = 0.5
        seq_reason = ""
    else:
        sequence_score, seq_reason = check_sequence(
            response_text, task_name, stage_index
        )
        if seq_reason:
            reason_parts.append(seq_reason)

    # ── 4. TRI-PARTITE FORMULA ───────────────────────────────────────────────
    tripartite_score = (
        0.4 * intent_score
        + 0.4 * completeness_score
        + 0.2 * sequence_score
    )

    # Stall penalty
    if is_stalling and not stage_advanced:
        tripartite_score = max(0.0, tripartite_score - 0.10)
        reason_parts.append("Warning: stalling — not advancing stage.")

    # Clamp
    tripartite_score = min(1.0, max(0.0, tripartite_score))

    # rule_score = the pure tripartite score (stage-machine fidelity signal).
    # This reflects ONLY intent classification + completeness + sequence.
    # It does NOT include confidence bonuses, so it truly varies per step.
    # Round to 2 decimal places to avoid floating-point artifacts in the API output.
    rule_score = round(tripartite_score, 2)

    # --- v8 PATCH: CONFIDENCE-AWARE REWARD ADJUSTMENTS (Steps 3-10) ---
    # step_reward starts from the tripartite base and is then tuned by the
    # confidence-aware intent extractor. It is the RL training signal.
    step_reward = tripartite_score

    if detected_intents_dict:
        penalties_log: List[str] = []
        bonuses_log:   List[str] = []

        # ── STEP 4: Soft penalties with floor protection ──────────────────
        # STEP 4a: Repetitive penalty
        if is_repetitive:
            step_reward -= 0.15
            penalties_log.append("repetitive:-0.15")

        # STEP 4b: Stalling penalty
        if is_stalling and stall_count > 0:
            stall_pen = 0.1 * stall_count
            step_reward -= stall_pen
            penalties_log.append(f"stalling:-{stall_pen:.2f}")

        # ── Missing required intents penalty ──────────────────────────────
        reqs = TASK_REQUIREMENTS.get(task_name, {}).get("required", [])
        for req_intent in reqs:
            intent_data = detected_intents_dict.get(req_intent, {})
            if not intent_data.get("present", False):
                step_reward -= 0.15
                penalties_log.append(f"missing_{req_intent}:-0.15")

        # ── Positive bonuses (confidence-weighted) ────────────────────────
        for req_intent in reqs:
            intent_data = detected_intents_dict.get(req_intent, {})
            if intent_data.get("present", False):
                bonus = 0.15 * intent_data.get("confidence", 1.0)
                step_reward += bonus
                bonuses_log.append(f"{req_intent}:+{bonus:.2f}")

        # ── Closure bonus (confidence-weighted, near-end only) ────────────
        closure_data = detected_intents_dict.get("closure", {})
        if closure_data.get("present", False) and stage_name == "closure":
            bonus = 0.1 * closure_data.get("confidence", 1.0)
            step_reward += bonus
            bonuses_log.append(f"closure:+{bonus:.2f}")

        # ── Soft floor (non-off-topic actions get at least 0.1) ──────────
        off_topic_data = detected_intents_dict.get("off_topic", {})
        if not off_topic_data.get("present", False):
            step_reward = max(step_reward, 0.1)

        # ── Off-topic — confidence-gated penalty ──────────────────────────
        if off_topic_data.get("present", False):
            off_conf = off_topic_data.get("confidence", 0.0)
            if off_conf > 0.8:
                step_reward = 0.0
                penalties_log.append(f"off_topic_hard(conf={off_conf:.2f}):->0.0")
            else:
                step_reward = min(step_reward, 0.2)
                penalties_log.append(f"off_topic_soft(conf={off_conf:.2f}):cap_0.2")

        # ── Critical empathy check ────────────────────────────────────────
        if "empathy" in reqs:
            empathy_data = detected_intents_dict.get("empathy", {})
            if stage_name == "start" and not empathy_data.get("present", False):
                step_reward -= 0.25
                penalties_log.append("missing_empathy_at_start:-0.25")
                if not off_topic_data.get("present", False):
                    step_reward = max(step_reward, 0.1)

        # ── Protect valid critical actions ────────────────────────────────
        escalation_data = detected_intents_dict.get("escalation", {})
        refund_data     = detected_intents_dict.get("refund", {})
        replace_data    = detected_intents_dict.get("replacement", {})

        if escalation_data.get("present") and escalation_data.get("confidence", 0) > 0.7:
            step_reward = max(step_reward, 0.5)
            bonuses_log.append("escalation_guard:min_0.5")
        if refund_data.get("present") and refund_data.get("confidence", 0) > 0.7:
            step_reward = max(step_reward, 0.6)
            bonuses_log.append("refund_guard:min_0.6")
        if replace_data.get("present") and replace_data.get("confidence", 0) > 0.7:
            step_reward = max(step_reward, 0.6)
            bonuses_log.append("replacement_guard:min_0.6")

        # ── Stage progression reward (+0.15 additive) ─────────────────────
        expected_flow = EXPECTED_FLOW.get(task_name, [])
        if stage_name and stage_advanced and len(expected_flow) > 1:
            try:
                current_idx = expected_flow.index(stage_name)
                if current_idx > 0:  # not the very first stage
                    step_reward += 0.15
                    bonuses_log.append("correct_stage_progression:+0.15")
            except ValueError:
                pass

        # ── Partial credit for empathy/useful-info ────────────────────────
        empathy_d = detected_intents_dict.get("empathy", {})
        apology_d = detected_intents_dict.get("apology", {})
        if empathy_d.get("present") or apology_d.get("present"):
            step_reward += 0.05
            bonuses_log.append("empathy_or_apology:+0.05")

        tracking_d = detected_intents_dict.get("tracking_info", {})
        if tracking_d.get("present") or refund_data.get("present") or replace_data.get("present"):
            step_reward += 0.10
            bonuses_log.append("useful_info:+0.10")

        # ── Normalize step_reward ─────────────────────────────────────────
        step_reward = max(0.01, min(0.99, step_reward))

    else:
        step_reward = tripartite_score
        penalties_log = []
        bonuses_log = []

    # ── 5. REWARD SHAPER (internal post-processor) ────────────────────────────
    # RewardShaper applies additional penalty/bonus rules without changing schema.
    is_ordered = True
    if StageSequenceGuard and history_intents is not None:
        guard_res = StageSequenceGuard.check_sequence(
            task_name, stage_name, intents_set, history_intents
        )
        is_ordered = guard_res.is_ordered

    # We call our new shape_reward (v4 internal logic)
    step_reward = shape_reward(
        base_reward=step_reward,
        is_repetitive=is_repetitive,
        is_stalling=is_stalling,
        stall_count=stall_count,
        intents=intents_set,
        stage_name=stage_name,
        stage_advanced=stage_advanced,
        detected_intents_dict=detected_intents_dict,
        task_name=task_name,
        customer_interaction_data=None,
        last_reward=last_reward,
        is_ordered=is_ordered,
        low_reward_streak=low_reward_streak,
    )


    # ── 6. BUILD REASON ──────────────────────────────────────────────────────
    if not reason_parts:
        if completeness_score >= 0.9 and intent_score >= 0.9:
            reason = "Correct and complete response."
        elif completeness_score >= 0.5:
            reason = "Response follows correct intent."
        else:
            reason = "Response partially addresses customer needs."
    else:
        reason = " ".join(reason_parts)

    return round(step_reward, 2), round(rule_score, 2), round(completeness_score, 2), round(sequence_score, 2), reason


def _build_reward_reason(
    success: bool,
    closure_reached: bool,
    failure_reason: str,
) -> str:
    """
    Build a human-readable explanation of the episode reward.
    """
    if success and closure_reached:
        return "Resolved with closure"
    if success:
        return "Resolved without formal closure"
    reason_map = {
        "consecutive_failures": "Too many consecutive wrong responses",
        "repetition_limit":     "Repeated responses exceeded limit",
        "max_steps_exceeded":   "Maximum steps reached without resolution",
        "unresolved":           "Issue was not resolved",
    }
    return reason_map.get(failure_reason, f"Unresolved: {failure_reason}")


def _determine_failure_reason(episode: "EpisodeState") -> str:
    """Determine why the episode ended without full success."""
    if episode.consecutive_failures >= episode.max_consecutive_failures:
        return "consecutive_failures"
    if episode.consecutive_repetitions >= episode.max_consecutive_repetitions:
        return "repetition_limit"
    if episode.steps_taken >= episode.max_steps:
        return "max_steps_exceeded"
    return "unresolved"


# ===========================================================================
# CUSTOMER RESPONSE GENERATION
# ===========================================================================

def _get_customer_response(
    task_name: str,
    stage_name: str,
    mood: str,
    script_index: int,
) -> str:
    """Return a scripted customer response for the given stage + mood."""
    try:
        variants = CUSTOMER_SCRIPTS[task_name][stage_name][mood]
        return variants[script_index % len(variants)]
    except KeyError:
        try:
            variants = CUSTOMER_SCRIPTS[task_name][stage_name]["neutral"]
            return variants[script_index % len(variants)]
        except KeyError:
            return "Is there anything else you can do for me?"


def _get_done_message(resolved: bool, mood: str) -> str:
    """Return a terminal customer message."""
    if resolved:
        if mood == "satisfied":
            return "Wonderful! Thank you so much for your help. My issue is fully resolved!"
        elif mood == "neutral":
            return "Thank you for your help. My issue has been resolved."
        else:
            return "Fine. I suppose that resolves things. Thank you."
    else:
        return "I'm still not satisfied. I'll need to contact you again or escalate further."


# ===========================================================================
# MOOD EVOLUTION (relaxed — accepted intents no longer worsen mood)
# ===========================================================================

def _evolve_mood(
    current_mood: str,
    intents: List[str],
    stage_advanced: bool,
    stage_accepted: bool,
) -> str:
    """
    Compute the customer's new mood after the agent's response.

    Rules:
    - Stage advance with strong resolution/empathy intent → mood improves
    - Any stage advance → mood at least holds
    - Accepted (but not advancing) → mood holds
    - Not accepted AND off_topic → mood worsens
    """
    mood_ladder = ["angry", "neutral", "satisfied"]
    idx = mood_ladder.index(current_mood)
    intents_set = set(intents)

    if stage_advanced:
        # Positive intents cause mood to improve more reliably
        if intents_set & {"resolution_offer", "confirmation", "de_escalation"}:
            idx = min(idx + 1, 2)
        else:
            # Neutral advance — mood holds or slightly improves
            idx = min(idx + 1, 2)
    elif stage_accepted:
        # Accepted but not advancing — mood holds (no change)
        pass
    else:
        # Not accepted
        if "off_topic" in intents and len(intents) == 1:
            # Truly off_topic → mood worsens
            idx = max(idx - 1, 0)
        # Wrong-but-real-intent: hold mood (don't punish twice)

    return mood_ladder[idx]


# ===========================================================================
# ENVIRONMENT
# ===========================================================================

class CustomerSupportEnvironment(Environment):
    """
    Multi-step, stateful Customer Support Resolution Environment (v5 — Normalized Reward).

    Key properties:
    - Multi-intent classification (agent responses carry multiple intents)
    - Partial credit rewards (graded tiers instead of binary)
    - Stage-skip for strong responses (skip minor stages)
    - Relaxed failure thresholds (4 failures, 3 repetitions)
    - Pure rule-based grader (no LLM calls, fully deterministic)
    - Normalized reward: always in [0.0, 1.0]
    - reward == grader_score at episode end (perfect alignment)
    - reward_reason explains the final score in plain language
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task: Optional[Dict[str, Any]] = None
        self._task_name: str = ""
        self._conversation_history: List[Dict[str, str]] = []
        self._episode: Optional[EpisodeState] = None
        self._script_index: int = 0

    # -----------------------------------------------------------------------
    def reset(
        self,
        task_name: Optional[str] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> CustomerSupportObservation:
        """
        Reset the environment and start a new episode.

        Args:
            task_name: 'order_status' | 'damaged_product' | 'escalation'.
                       Defaults to 'order_status'.
            episode_id: Optional episode ID override.
        """
        # Resolve platform-required IDs to internal task names
        TASK_ALIASES = {
            "task_easy":   "order_status",
            "task_medium": "damaged_product",
            "task_hard":   "escalation",
        }
        if task_name in TASK_ALIASES:
            task_name = TASK_ALIASES[task_name]

        if task_name is None or task_name not in TASKS:
            task_name = "order_status"


        self._task_name = task_name
        self._task = TASKS[task_name]
        stages = TASK_STAGES[task_name]
        initial_mood = self._task["initial_mood"]

        self._conversation_history = []
        self._script_index = 0

        self._episode = EpisodeState(
            task_name=task_name,
            stages=stages,
            stage_index=0,
            customer_mood=initial_mood,
            issue_status="unresolved",
            steps_taken=0,
            max_steps=self._task["max_steps"],
            max_consecutive_failures=self._task.get("max_consecutive_failures", 4),
            max_consecutive_repetitions=self._task.get("max_consecutive_repetitions", 3),
        )

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        initial_message = _get_customer_response(
            task_name, "start", initial_mood, script_index=0
        )
        self._conversation_history.append({"role": "user", "content": initial_message})

        hints: List[str] = []
        if self._task.get("provide_hints") and stages[0].get("hint"):
            hints = [stages[0]["hint"]]

        return CustomerSupportObservation(
            customer_message=initial_message,
            conversation_history=list(self._conversation_history),
            task_name=task_name,
            task_difficulty=self._task["difficulty"],
            step=0,
            max_steps=self._task["max_steps"],
            is_resolved=False,
            conversation_stage="start",
            customer_mood=initial_mood,
            issue_status="unresolved",
            intent_detected="",
            intents_detected=[],
            hints=hints,
            rule_score=0.0,
            grader_score=0.0,
            done=False,
            reward=0.0,
            reward_reason="",
            success=False,
            repetition_count=0,
            stall_count=0,
            failure_reason="",
            task_context=self._task.get("context"),
        )

    # -----------------------------------------------------------------------
    def step(self, action: CustomerSupportAction) -> CustomerSupportObservation:  # type: ignore[override]
        """
        Process the agent's response and advance the episode state (flexible state machine).
        """
        # Guard: not reset yet
        if self._task is None or self._episode is None:
            return CustomerSupportObservation(
                customer_message="Please reset the environment first.",
                task_name="",
                task_difficulty="easy",
                step=0,
                max_steps=10,
                is_resolved=False,
                done=True,
                reward=0.0,
                success=False,
                repetition_count=0,
                stall_count=0,
                failure_reason="not_reset",
                intents_detected=[],
            )

        ep = self._episode
        ep.steps_taken += 1
        self._state.step_count += 1

        agent_response = action.response

        # Record agent turn in history
        self._conversation_history.append({"role": "assistant", "content": agent_response})

        # ── 1. MULTI-INTENT CLASSIFICATION (v7 Patch) ──────────────────────
        # Use our new rule-based intent extractor (v7 patch)
        detected_intents = extract_intents(agent_response, self._task_name)
        
        # Bridge to the original intent names for the state machine
        intents = get_bridge_intents(detected_intents)
        if not intents:
            intents = ["off_topic"]
            
        intents_set: Set[str] = set(intents)
        primary_intent: str = intents[0]

        # ── 2. TRACK APOLOGY (damaged_product only) ──────────────────────────
        if "apology" in intents_set and not ep.had_apology:
            ep.had_apology = True

        # ── 3. REPETITION DETECTION (relaxed Jaccard 0.85) ───────────────────
        rep_flag, rep_reason = _is_repetitive(
            agent_response,
            intents,
            ep.intent_history,
            ep.response_history,
            ep.response_fingerprints,
        )
        if rep_flag:
            ep.consecutive_repetitions += 1
        else:
            ep.consecutive_repetitions = 0

        # Record for next-step detection
        ep.intent_history.append(intents)
        ep.response_history.append(agent_response)
        ep.response_fingerprints.append(_response_fingerprint(agent_response))

        # ── 4. STAGE MACHINE ADVANCE ──────────────────────────────────────────
        current_stage = ep.current_stage
        accepted_intents: Set[str] = current_stage["accepted_intents"]
        advance_intents: Set[str] = current_stage["advance_intents"]
        max_stall: int = current_stage.get("max_stall", 2)

        # Multi-intent gate: at least ONE detected intent must be in accepted set
        stage_accepted = bool(intents_set & accepted_intents)
        # Advance gate: at least ONE detected intent must be in advance set
        can_advance = bool(intents_set & advance_intents)
        stage_advanced = False

        # Check stage-skip first (strong responses can skip minor stages)
        skip_idx = None
        if not ep.at_final_stage and not rep_flag and can_advance:
            skip_idx = _get_skip_target(
                self._task_name, intents_set, ep.stage_index, ep.stages
            )

        if skip_idx is not None:
            ep.stage_index = skip_idx
            stage_advanced = True
            ep.issue_status = "in_progress"
            ep.consecutive_failures = 0
            ep.stall_steps = 0
        elif can_advance and not ep.at_final_stage and not rep_flag:
            ep.stage_index += 1
            stage_advanced = True
            ep.issue_status = "in_progress"
            ep.consecutive_failures = 0
            ep.stall_steps = 0
        else:
            # No advance — update stall and failure counters
            if not stage_advanced:
                ep.stall_steps += 1

            if not stage_accepted:
                ep.consecutive_failures += 1
            else:
                # Accepted but didn't advance (e.g. at_final_stage, or no advance intent)
                ep.consecutive_failures = 0

        # ── 5. STALL DETECTION ───────────────────────────────────────────────
        is_stalling = ep.stall_steps > max_stall

        # ── 6. RESOLUTION CHECK ──────────────────────────────────────────────
        # Resolve at final stage with advance intent, OR mood-based resolution
        if ep.at_final_stage and can_advance and not rep_flag:
            ep.resolved = True
            ep.closure_reached = True
            ep.issue_status = "resolved"
            ep.customer_mood = _evolve_mood(ep.customer_mood, intents, True, True)
        else:
            ep.customer_mood = _evolve_mood(ep.customer_mood, intents, stage_advanced, stage_accepted)

        # ── 7. COMPUTE STEP REWARD (tri-partite: intent + completeness + sequence) ──
        step_reward, rule_score, completeness_score, sequence_score, step_reason = (
            _compute_step_reward(
                response_text=agent_response,
                intents=intents,
                intents_set=intents_set,
                accepted_intents=accepted_intents,
                stage_advanced=stage_advanced,
                is_repetitive=rep_flag,
                is_stalling=is_stalling,
                prev_step_was_wrong=ep.prev_step_was_wrong,
                task_name=self._task_name,
                stage_name=ep.stage_name,
                stage_index=ep.stage_index,
                detected_intents_dict=detected_intents,
                history_intents=[set(h) for h in ep.intent_history],
                last_reward=ep.trajectory[-1]["reward"] if ep.trajectory else 1.0,
                low_reward_streak=ep.low_reward_streak,
            )
        )

        # Update streak counter
        if step_reward < 0.3:
            ep.low_reward_streak += 1
        else:
            ep.low_reward_streak = 0

        # Track for next-step recovery
        ep.prev_step_was_wrong = (not stage_accepted and not rep_flag)

        # Internal failure recovery flag — used by inference.py to force recovery prompt
        # Does NOT affect any API output or schema fields.
        ep.failure_recovery_active = (step_reward < 0.3)


        # ── 8. RECORD TRAJECTORY STEP ─────────────────────────────────────────
        ep.trajectory.append({
            "step":               ep.steps_taken,
            "intent":             primary_intent,
            "intents":            intents,
            "stage":              ep.stage_name,
            "stage_accepted":     stage_accepted,
            "stage_advanced":     stage_advanced,
            "is_repetitive":      rep_flag,
            "is_stalling":        is_stalling,
            "rule_score":         rule_score,
            "completeness_score": completeness_score,
            "sequence_score":     sequence_score,
            "reward":             step_reward,
            "mood":               ep.customer_mood,
            "step_reason":        step_reason,
            "detected_intents":   detected_intents,
        })

        # ── 9. CHECK ALL FAILURE / DONE CONDITIONS ────────────────────────────
        is_final_step = ep.done

        # ── 10. GRADER + NORMALIZED REWARD (computed at episode end) ──────────
        grader_score = 0.0
        success = False
        failure_reason = ""
        reward_reason = ""

        if is_final_step:
            # Determine success and failure reason
            success = ep.resolved  # resolved (with or without closure) counts as success
            if ep.resolved and ep.closure_reached:
                failure_reason = ""
            elif ep.resolved:
                failure_reason = ""
            else:
                failure_reason = _determine_failure_reason(ep)
            ep.failure_reason = failure_reason

            # No-apology trajectory penalty for damaged_product tasks
            if (
                self._task_name == "damaged_product"
                and ep.steps_taken >= self._task.get("no_apology_by_step", 3)
                and not ep.had_apology
            ):
                # Retroactively reduce avg rule_score in trajectory by a small factor
                for t in ep.trajectory:
                    t["rule_score"] = max(0.0, t["rule_score"] - 0.05)

            # Pure rule-based grader — deterministic, no LLM
            required_intents = TASK_REQUIREMENTS.get(self._task_name, {}).get("required", [])
            grader_score = grade_episode(
                trajectory=ep.trajectory,
                final_stage=ep.stage_name,
                final_mood=ep.customer_mood,
                resolved=ep.resolved,
                closure_reached=ep.closure_reached,
                steps_taken=ep.steps_taken,
                max_steps=ep.max_steps,
                step_rewards=[t["reward"] for t in ep.trajectory],
                required_intents=required_intents,
            )

            reward_reason = _build_reward_reason(success, ep.closure_reached, failure_reason)

        # ── Normalized reward ─────────────────────────────────────────────────
        # Final step: reward = grader_score (perfect alignment, bounded [0,1])
        # Other steps: reward = max(0.0, step_reward) — small positive signal only
        if is_final_step:
            reward_out = max(0.01, min(0.99, grader_score))
        else:
            reward_out = max(0.01, min(0.99, step_reward))

        # ── STEP 2: Accumulate confirmed intents (confidence > 0.6) ───────────
        for intent_name, intent_data in detected_intents.items():
            if intent_data.get("present", False) and intent_data.get("confidence", 0.0) > 0.6:
                if intent_name in ep.collected_intents:
                    ep.collected_intents[intent_name] = True

        # ── STEPS 3-7: Required-intent reward cap (constraint layer) ──────────
        # This is a POST-PROCESSING cap only. It does NOT change reward logic;
        # it prevents high rewards when critical intents were never addressed.
        _required_intents = TASK_REQUIREMENTS.get(self._task_name, {}).get("required", [])
        _missing_required = any(
            not ep.collected_intents.get(intent, False)
            for intent in _required_intents
        )

        # STEP 5: Only cap if reward is already high (> threshold) — does NOT
        # penalize low/mid rewards from agents that are still progressing.
        if is_final_step:
            # STEP 4: Stricter cap at final evaluation
            if _missing_required and reward_out > 0.5:
                reward_out = 0.5
        else:
            # STEP 3: Soft cap during episode
            if _missing_required and reward_out > 0.6:
                reward_out = 0.6

        # Debug: Uncomment to trace reward caps during dev
        # print(f"[DEBUG] Missing={_missing_required} → reward_out={reward_out:.3f}")

        # ── 11. NEXT CUSTOMER MESSAGE ──────────────────────────────────────────
        self._script_index += 1
        if is_final_step:
            next_message = _get_done_message(ep.resolved, ep.customer_mood)
        else:
            next_message = _get_customer_response(
                self._task_name, ep.stage_name, ep.customer_mood, self._script_index
            )
            self._conversation_history.append({"role": "user", "content": next_message})

        # ── 12. HINTS (easy task only, not at final step) ─────────────────────
        hints: List[str] = []
        if self._task.get("provide_hints") and not is_final_step:
            hint_text = ep.current_stage.get("hint")
            if hint_text:
                hints = [hint_text]

        # Final failure reason assignment
        out_failure_reason = ""
        if is_final_step:
            out_failure_reason = failure_reason
        elif rep_flag:
            out_failure_reason = "Repetitive content detected"
        elif is_stalling:
            out_failure_reason = "Warning: Approaching stall limit"

        # Convert confidence-aware dict → plain bool dict for the schema-validated field.
        # The full confidence dict (detected_intents) is stored in trajectory for
        # internal reward use; the API-facing 'intents' field stays Dict[str, bool].
        intents_bool: Dict[str, bool] = {
            k: bool(v.get("present", False))
            for k, v in detected_intents.items()
        }

        return CustomerSupportObservation(
            customer_message=next_message,
            conversation_history=list(self._conversation_history),
            task_name=self._task_name,
            task_difficulty=self._task["difficulty"],
            step=ep.steps_taken,
            max_steps=ep.max_steps,
            is_resolved=ep.resolved,
            conversation_stage=ep.stage_name,
            customer_mood=ep.customer_mood,
            issue_status=ep.issue_status,
            intent_detected=primary_intent,
            intents_detected=intents,
            intents=intents_bool,
            hints=hints,
            rule_score=rule_score,
            grader_score=grader_score,
            done=is_final_step,
            reward=reward_out,
            reward_reason=reward_reason if is_final_step else step_reason,
            success=success,
            repetition_count=ep.consecutive_repetitions,
            stall_count=ep.stall_steps,
            failure_reason=out_failure_reason,
            task_context=self._task.get("context"),
        )

    # -----------------------------------------------------------------------
    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state
