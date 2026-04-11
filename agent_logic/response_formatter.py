"""
response_formatter.py — Output post-processor for BPO OpenEnv agent responses.

Responsibilities:
  1. Detect Action / Stage / Status signals from response text.
  2. Remove conversational question patterns (medium task fix).
  3. Light cleaning: deduplicate sentences, trim whitespace.
  4. Append a minimal structured block to every response.

STRICT SCOPE: formatting only — no scoring, reward, env, or API schema changes.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# 1. Signal Detection
# ---------------------------------------------------------------------------

def detect_action(text: str) -> str:
    """Map response text to an action label via simple keyword matching."""
    lower = text.lower()
    if "tracking" in lower or "track" in lower:
        return "ProvideTracking"
    if "replacement" in lower or "replace" in lower:
        return "Replacement"
    if "refund" in lower:
        return "Refund"
    if "escalate" in lower or "manager" in lower:
        return "Escalation"
    return "GeneralSupport"


def detect_stage(done: bool, step: int) -> str:
    """Infer the conversation stage from episode state."""
    if done:
        return "Closure"
    if step == 1:
        return "Inquiry"
    return "Resolution"


def detect_status(done: bool) -> str:
    """Return a binary processing status."""
    return "Completed" if done else "Processing"


# ---------------------------------------------------------------------------
# 2. Medium-task fix — inject resolution prefix + strip questions
# ---------------------------------------------------------------------------

_REMOVE_PATTERNS: list[str] = [
    r"please provide[^.!?]*[.!?]?",
    r"could you[^.!?]*[.!?]?",
    r"would you like[^.!?]*[.!?]?",
    r"let me know[^.!?]*[.!?]?",
]

_REMOVE_RE = re.compile(
    "|".join(_REMOVE_PATTERNS),
    flags=re.IGNORECASE,
)


def _strip_question_sentences(text: str) -> str:
    """Remove sentences that contain passive/question patterns."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    cleaned: list[str] = []
    for sentence in sentences:
        if _REMOVE_RE.search(sentence):
            continue
        cleaned.append(sentence)
    return " ".join(cleaned).strip()


def _apply_medium_fix(response: str, task_name: str) -> str:
    """For task_medium: inject resolution prefix if none present."""
    if task_name != "task_medium":
        return response
    lower = response.lower()
    if "replacement" not in lower and "refund" not in lower:
        response = "I've initiated a replacement for your product. " + response
    return response


# ---------------------------------------------------------------------------
# 3. Light cleaning — dedup sentences, trim whitespace, max 3 sentences
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    - Remove duplicate sentences (case-insensitive).
    - Strip question sentences (passive / question patterns).
    - Remove existing structured tag blocks so we never duplicate them.
    - Collapse extra whitespace.
    - Keep at most 3 sentences.
    """
    # Strip any pre-existing structured block (guard against double-appending)
    text = re.sub(
        r"\n*\[Action:[^\]]*\]\n\[Stage:[^\]]*\]\n\[Status:[^\]]*\]",
        "",
        text,
    ).strip()

    # Strip question sentences
    text = _strip_question_sentences(text)

    # Split, dedup, limit to 3
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    seen: set[str] = set()
    deduped: list[str] = []
    for s in sentences:
        key = s.lower().strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(s)
    deduped = deduped[:3]

    # Collapse whitespace
    result = " ".join(deduped)
    result = re.sub(r"[ \t]{2,}", " ", result)
    return result.strip()


# ---------------------------------------------------------------------------
# 4. Master formatter
# ---------------------------------------------------------------------------

def format_response(
    response: str,
    step: int,
    done: bool,
    task_name: str,
) -> str:
    """
    Full post-processing pipeline.

    Steps:
      1. Medium-task resolution prefix injection.
      2. Light cleaning (dedup, strip question sentences, trim).
      3. Detect Action / Stage / Status.
      4. Append structured block exactly once.
    """
    # Stage 1 — medium task fix
    response = _apply_medium_fix(response, task_name)

    # Stage 2 — clean
    response = clean_text(response)

    # Stage 3 — detect signals
    action = detect_action(response)
    stage  = detect_stage(done, step)
    status = detect_status(done)

    # Stage 4 — append structured block (single trailing newline before block)
    structured = (
        f"\n\n[Action: {action}]\n"
        f"[Stage: {stage}]\n"
        f"[Status: {status}]"
    )

    return response.strip() + structured
