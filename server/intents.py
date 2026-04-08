"""
Intent Extractor — v8 (Confidence-aware, Precision-first)

Returns confidence-scored intent dicts instead of plain booleans.
Satisfies Steps 1 & 2 of the reward-tuning patch.

Schema:
    {
        "intent_name": {"present": bool, "confidence": float}
    }
"""

import re
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# STEP 1 + STEP 2: Confidence-aware, precision-first intent extractor
# ---------------------------------------------------------------------------

def extract_intents(text: str, task: str) -> Dict[str, Dict]:
    """
    Extract intents from text using strict, precision-first keyword rules.

    Returns a dict of the form:
        {
          "intent_name": {"present": bool, "confidence": float}
        }

    Rules deliberately prefer precision over recall to reduce noisy signals.
    """
    lower_text = text.lower()

    # ── empathy ──────────────────────────────────────────────────────────────
    empathy_kws = ["understand", "frustration", "empath", "appreciate",
                   "hear you", "i can see", "that must"]
    empathy_hits = sum(1 for kw in empathy_kws if kw in lower_text)
    empathy_present = empathy_hits >= 1
    empathy_conf = min(1.0, 0.4 + 0.2 * empathy_hits)

    # ── apology ──────────────────────────────────────────────────────────────
    apology_kws = ["sorry", "apologize", "apology", "regret", "apologies"]
    apology_hits = sum(1 for kw in apology_kws if kw in lower_text)
    apology_present = apology_hits >= 1
    apology_conf = min(1.0, 0.5 + 0.2 * apology_hits)

    # ── tracking_info — STEP 2: strict dual condition ──────────────────────
    # ONLY true if: contains tracking number pattern AND contains "tracking"
    has_tracking_word = "tracking" in lower_text
    has_tracking_number = bool(
        re.search(r"TRK\d+", text) or re.search(r"\b\d{5,}\b", text)
    )
    tracking_present = has_tracking_word and has_tracking_number
    if tracking_present:
        tracking_conf = 0.9
    elif has_tracking_word:
        tracking_conf = 0.3  # word present but no number → low confidence
        tracking_present = False
    else:
        tracking_conf = 0.05

    # ── delivery_info — STEP 2: strict dual condition ─────────────────────
    # ONLY true if: contains date-like token AND delivery keywords
    delivery_kws = ["arrive", "delivery", "expected"]
    has_delivery_keyword = any(kw in lower_text for kw in delivery_kws)
    # Date-like token: month names, day+ordinal, "days", "week" patterns
    _DATE_PATTERN = re.compile(
        r"\b(\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4}|"          # 01/02/2026
        r"(january|february|march|april|may|june|july|"
        r"august|september|october|november|december)"
        r"|\d{1,2}(st|nd|rd|th)?"                         # 3rd, 5th
        r"|\d+\s*(business\s*)?days?)\b",
        re.IGNORECASE,
    )
    has_date_token = bool(_DATE_PATTERN.search(text))
    delivery_present = has_delivery_keyword and has_date_token
    if delivery_present:
        delivery_conf = 0.88
    elif has_delivery_keyword:
        delivery_conf = 0.2
        delivery_present = False
    else:
        delivery_conf = 0.05

    # ── refund ───────────────────────────────────────────────────────────────
    refund_kws = ["refund", "money back", "credit your account", "full refund"]
    refund_hits = sum(1 for kw in refund_kws if kw in lower_text)
    refund_present = refund_hits >= 1
    refund_conf = min(1.0, 0.6 + 0.15 * refund_hits)

    # ── replacement ──────────────────────────────────────────────────────────
    replacement_kws = ["replace", "replacement", "new unit", "send a new", "ship a new"]
    replacement_hits = sum(1 for kw in replacement_kws if kw in lower_text)
    replacement_present = replacement_hits >= 1
    replacement_conf = min(1.0, 0.6 + 0.15 * replacement_hits)

    # ── escalation — STEP 2: strict keywords only ────────────────────────
    # ONLY true if: contains ["manager", "supervisor", "escalate"]
    escalation_strict_kws = ["manager", "supervisor", "escalate"]
    escalation_hits = sum(1 for kw in escalation_strict_kws if kw in lower_text)
    escalation_present = escalation_hits >= 1
    escalation_conf = min(1.0, 0.5 + 0.25 * escalation_hits)

    # ── closure ──────────────────────────────────────────────────────────────
    closure_kws = ["anything else", "have a great day", "thank you", "take care",
                   "have a wonderful day", "all set", "you're all set"]
    closure_hits = sum(1 for kw in closure_kws if kw in lower_text)
    closure_present = closure_hits >= 1
    closure_conf = min(1.0, 0.5 + 0.2 * closure_hits)

    # ── information_request ───────────────────────────────────────────────────
    info_req_kws = ["please provide", "confirm", "could you", "what is",
                    "can you provide", "please share", "order number"]
    info_req_hits = sum(1 for kw in info_req_kws if kw in lower_text)
    info_req_present = info_req_hits >= 1
    info_req_conf = min(1.0, 0.4 + 0.15 * info_req_hits)

    # ── off_topic — STEP 2: STRICT match only ────────────────────────────
    # ONLY if contains exact phrases about redirecting to website
    off_topic_strict = ["visit our website", "check website", "go to website",
                        "check our website", "online portal"]
    off_topic_hits = sum(1 for phrase in off_topic_strict if phrase in lower_text)
    off_topic_present = off_topic_hits >= 1
    # High confidence only if an exact match exists
    off_topic_conf = 0.9 if off_topic_present else 0.0

    return {
        "empathy":            {"present": empathy_present,      "confidence": empathy_conf},
        "apology":            {"present": apology_present,      "confidence": apology_conf},
        "tracking_info":      {"present": tracking_present,     "confidence": tracking_conf},
        "delivery_info":      {"present": delivery_present,     "confidence": delivery_conf},
        "refund":             {"present": refund_present,       "confidence": refund_conf},
        "replacement":        {"present": replacement_present,  "confidence": replacement_conf},
        "escalation":         {"present": escalation_present,   "confidence": escalation_conf},
        "closure":            {"present": closure_present,      "confidence": closure_conf},
        "information_request":{"present": info_req_present,     "confidence": info_req_conf},
        "off_topic":          {"present": off_topic_present,    "confidence": off_topic_conf},
    }


def get_bridge_intents(detected_intents: Dict[str, Dict]) -> List[str]:
    """
    Maps confidence-scored intents to original intent names used by the stage machine.

    Reads the new {"present": bool, "confidence": float} format.
    Only includes an intent if present=True.
    """
    mapping: List[Tuple[str, str]] = [
        ("apology",             "apology"),
        ("empathy",             "de_escalation"),
        ("tracking_info",       "information_provide"),
        ("delivery_info",       "information_provide"),
        ("refund",              "resolution_offer"),
        ("replacement",         "resolution_offer"),
        ("escalation",          "resolution_offer"),
        ("information_request", "information_request"),
        ("closure",             "confirmation"),
    ]

    bridged = []
    for new_id, old_id in mapping:
        intent_data = detected_intents.get(new_id, {})
        if intent_data.get("present", False):
            if old_id not in bridged:
                bridged.append(old_id)

    # Add greeting when empathy or apology is detected
    empathy_data = detected_intents.get("empathy", {})
    apology_data = detected_intents.get("apology", {})
    if empathy_data.get("present") or apology_data.get("present"):
        if "greeting" not in bridged:
            bridged.append("greeting")

    return bridged


# ---------------------------------------------------------------------------
# Task 3.A: Mood Extractor (angry, confused, neutral)
# ---------------------------------------------------------------------------

def extract_mood(text: str) -> str:
    """
    Classify the customer's mood based on keyword patterns and punctuation.
    Used internally by inference.py for dynamic tone adjustment.
    """
    lower = text.lower()
    
    # ── Angry ────────────────────────────────────────────────────────────────
    angry_kws = ["terrible", "bad", "worst", "unacceptable", "disappointed", 
                 "annoyed", "ridiculous", "frustrated", "manager", "now!"]
    if any(kw in lower for kw in angry_kws) or "!!!" in text or text.isupper():
        return "angry"

    # ── Confused ─────────────────────────────────────────────────────────────
    confused_kws = ["how", "why", "where", "confused", "not sure", "don't know", 
                    "understand?", "explain", "help me understand"]
    if any(kw in lower for kw in confused_kws) or "?" in text:
        return "confused"

    return "neutral"
