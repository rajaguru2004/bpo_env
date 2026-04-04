"""
Customer Support Environment Implementation — v4 (Flexible Multi-Intent).

Simulates real-world customer support conversations with:
  - Multi-intent classification: a single response can carry multiple intents
  - Flexible stage transitions: any matching intent advances the stage
  - Stage-skip: strong resolution-oriented responses can skip minor stages
  - Partial credit rewards: graded rewards instead of binary pass/fail
  - Recovery mechanism: good step after bad step gets recovery bonus (+0.2)
  - Relaxed failure conditions: higher thresholds before termination
  - Hybrid grader: 0.85 * rule_score + 0.15 * llm_score
  - Deterministic step rewards (no LLM per step)
  - LLM judge called ONCE per episode for metadata + grader blend

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
except ImportError:
    from ..models import CustomerSupportAction, CustomerSupportObservation
    from .grader import grade_episode


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
    stall_steps: int = 0                    # steps without advancing current stage
    # Resolution tracking
    resolved: bool = False
    had_apology: bool = False
    closure_reached: bool = False
    # Recovery tracking
    prev_step_was_wrong: bool = False       # enables recovery bonus next step
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

# Step reward constants (v4 — reduced extremes, added partial tiers)
_R_CORRECT_ACTION  = +0.5   # response fully matches accepted_intents
_R_PARTIAL_ACTION  = +0.3   # some intents match (partial coverage)
_R_HELPFUL_ACTION  = +0.1   # off-stage but contextually helpful
_R_STAGE_ADVANCE   = +0.3   # triggered a stage transition
_R_RECOVERY_BONUS  = +0.2   # recovered from a wrong step
_R_WRONG_ACTION    = -0.3   # no intent matches accepted (was -0.4)
_R_REPETITION      = -0.5   # repeated response (unchanged)
_R_STALL           = -0.2   # stuck in stage too long (was -0.3)
_R_OFF_TOPIC       = -0.3   # no keywords detected (was -0.4)

# Terminal reward constants
_R_SUCCESS         = +1.0   # resolved AND closure reached
_R_PARTIAL_RESOLVE = +0.2   # resolved but no closure (was -0.5 — now positive!)
_R_FAILURE         = -0.5   # not resolved at episode end (was -1.0)
_R_NO_APOLOGY      = -0.3   # damaged_product: no apology — applied once at end (was -0.5)


def _compute_step_reward(
    intents: List[str],
    intents_set: Set[str],
    accepted_intents: Set[str],
    stage_advanced: bool,
    is_repetitive: bool,
    is_stalling: bool,
    prev_step_was_wrong: bool,
) -> Tuple[float, float]:
    """
    Compute step reward with partial credit and recovery.
    Returns (step_reward, rule_score_component [0,1]).
    """
    reward = 0.0

    if is_repetitive:
        reward += _R_REPETITION
    elif "off_topic" in intents and len(intents) == 1:
        reward += _R_OFF_TOPIC
    else:
        # Count how many detected intents are in the accepted set
        matching = intents_set & accepted_intents
        if len(matching) == 0:
            # No overlap at all — but not off_topic (has some real intent)
            reward += _R_WRONG_ACTION
        elif len(matching) >= 2 or (len(intents_set) == 1 and matching):
            # All (or near-all) intents match
            reward += _R_CORRECT_ACTION
        else:
            # Some intents match (partial)
            reward += _R_PARTIAL_ACTION

    # Stage advance bonus
    if stage_advanced:
        reward += _R_STAGE_ADVANCE

    # Recovery bonus: good response after a wrong one
    if prev_step_was_wrong and reward > 0 and not is_repetitive:
        reward += _R_RECOVERY_BONUS

    # Stall penalty (only if not advancing and not already penalized)
    if is_stalling and not stage_advanced and reward >= _R_WRONG_ACTION:
        reward += _R_STALL

    # Normalize rule_score to [0,1] for grader quality dimension
    # Range is roughly [-0.8, +1.0]; shift and scale
    rule_score = min(1.0, max(0.0, (reward + 1.0) / 2.0))

    return reward, rule_score


def _compute_terminal_reward(
    episode: "EpisodeState",
    task: Dict[str, Any],
) -> Tuple[float, bool, str]:
    """
    Compute terminal reward added only at the final step.
    Returns (terminal_reward, success, failure_reason).
    """
    if episode.resolved and episode.closure_reached:
        return _R_SUCCESS, True, ""

    if episode.resolved and not episode.closure_reached:
        # Partial success: resolved but didn't reach formal closure
        # Now positive to credit the resolution
        return _R_PARTIAL_RESOLVE, True, ""

    # Not resolved — determine why
    if episode.consecutive_failures >= episode.max_consecutive_failures:
        reason = "consecutive_failures"
    elif episode.consecutive_repetitions >= episode.max_consecutive_repetitions:
        reason = "repetition_limit"
    elif episode.steps_taken >= episode.max_steps:
        reason = "max_steps_exceeded"
    else:
        reason = "unresolved"

    return _R_FAILURE, False, reason


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
    Multi-step, stateful Customer Support Resolution Environment (v4 — Flexible).

    Key improvements over v3:
    - Multi-intent classification (agent responses carry multiple intents)
    - Partial credit rewards (graded tiers instead of binary)
    - Recovery mechanism (bonus for recovering from a wrong step)
    - Stage-skip for strong responses (skip minor stages)
    - Relaxed failure thresholds (4 failures, 3 repetitions)
    - Positive terminal reward for partial resolution
    - Hybrid grader: 0.85 * rule + 0.15 * llm
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
            llm_score=0.0,
            stage_reward=0.0,
            final_reward=0.0,
            grader_score=0.0,
            done=False,
            reward=0.0,
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

        # ── 1. MULTI-INTENT CLASSIFICATION ──────────────────────────────────
        intents: List[str] = classify_intents(agent_response)
        intents_set: Set[str] = set(intents)
        primary_intent: str = intents[0]  # highest-priority intent for compat

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

        # ── 7. COMPUTE STEP REWARD (partial credit + recovery) ───────────────
        step_reward, rule_score = _compute_step_reward(
            intents=intents,
            intents_set=intents_set,
            accepted_intents=accepted_intents,
            stage_advanced=stage_advanced,
            is_repetitive=rep_flag,
            is_stalling=is_stalling,
            prev_step_was_wrong=ep.prev_step_was_wrong,
        )

        # Track for next-step recovery
        ep.prev_step_was_wrong = (not stage_accepted and not rep_flag)

        stage_reward_val = _R_STAGE_ADVANCE if stage_advanced else 0.0

        # ── 8. RECORD TRAJECTORY STEP ─────────────────────────────────────────
        ep.trajectory.append({
            "step":           ep.steps_taken,
            "intent":         primary_intent,
            "intents":        intents,
            "stage":          ep.stage_name,
            "stage_accepted": stage_accepted,
            "stage_advanced": stage_advanced,
            "is_repetitive":  rep_flag,
            "is_stalling":    is_stalling,
            "rule_score":     rule_score,
            "reward":         step_reward,
            "mood":           ep.customer_mood,
        })

        # ── 9. CHECK ALL FAILURE / DONE CONDITIONS ────────────────────────────
        is_final_step = ep.done

        # ── 10. TERMINAL REWARD + GRADER (only at episode end) ────────────────
        llm_score = 0.0
        grader_score = 0.0
        terminal_reward = 0.0
        success = False
        failure_reason = ""

        # Damaged product: no-apology penalty (applied ONCE at final step only)
        no_apology_penalty = 0.0
        if (
            self._task_name == "damaged_product"
            and is_final_step
            and ep.steps_taken >= self._task.get("no_apology_by_step", 3)
            and not ep.had_apology
        ):
            no_apology_penalty = _R_NO_APOLOGY

        if is_final_step:
            terminal_reward, success, failure_reason = _compute_terminal_reward(ep, self._task)
            ep.failure_reason = failure_reason

            # LLM judge — contributes 15% to grader
            llm_score = _llm_judge_score(
                task_description=self._task["description"],
                conversation_history=self._conversation_history,
                final_response=agent_response,
            )

            # Update final step in trajectory with terminal reward and no-apology penalty
            ep.trajectory[-1]["reward"] = step_reward + terminal_reward + no_apology_penalty

            # Hybrid grader (0.85 rule + 0.15 llm)
            grader_score = grade_episode(
                trajectory=ep.trajectory,
                final_stage=ep.stage_name,
                final_mood=ep.customer_mood,
                resolved=ep.resolved,
                closure_reached=ep.closure_reached,
                steps_taken=ep.steps_taken,
                max_steps=ep.max_steps,
                step_rewards=[t["reward"] for t in ep.trajectory],
                llm_score=llm_score,
            )

        # Final reward = step reward + terminal reward + penalties (at end only)
        total_step_reward = (
            step_reward
            + (terminal_reward if is_final_step else 0.0)
            + (no_apology_penalty if is_final_step else 0.0)
        )

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
            hints=hints,
            rule_score=rule_score,
            llm_score=llm_score,
            stage_reward=stage_reward_val,
            final_reward=total_step_reward,
            grader_score=grader_score,
            done=is_final_step,
            reward=total_step_reward,
            success=success,
            repetition_count=ep.consecutive_repetitions,
            stall_count=ep.stall_steps,
            failure_reason=failure_reason if is_final_step else "",
            task_context=self._task.get("context"),
        )

    # -----------------------------------------------------------------------
    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state
