"""
Customer Support Environment Implementation — v2 (Multi-Step Stateful).

Simulates real-world customer support conversations with:
  - Stage-machine driven conversation flow (start → ... → closure)
  - Customer mood state (angry / neutral / satisfied) that evolves each step
  - Deterministic intent classification (zero LLM cost per step)
  - Dense intermediate rewards (stage advance, empathy, resolution quality)
  - LLM judge called ONCE per episode (end only)
  - Deterministic grader producing a final 0.0–1.0 episode score

Agent (LLM) acts as a customer support executive.
Three difficulty levels: easy / medium / hard.
"""

import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
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
# INTENT CLASSIFICATION (deterministic, zero LLM cost)
# ===========================================================================

_INTENT_KEYWORDS: Dict[str, List[str]] = {
    "apology": [
        "sorry", "apologize", "apologies", "sincerely apologize",
        "deeply sorry", "regret", "i understand your frustration",
        "i can see why", "you're right to be",
    ],
    "de_escalation": [
        "calm", "i hear you", "let me help", "i will personally",
        "take ownership", "i assure you", "i promise", "rest assured",
        "i completely understand", "totally understandable", "valid concern",
    ],
    "information_request": [
        "could you", "can you provide", "may i ask", "do you have",
        "what is your", "could you share", "please provide",
        "order number", "email address", "phone number",
    ],
    "resolution_offer": [
        "replacement", "replace", "refund", "send a new", "new unit",
        "ship out", "process your", "arrange", "compensation",
        "full refund", "credit", "re-send", "dispatch", "issue a",
        "initiate", "i will process",
    ],
    "escalation": [
        "escalate", "supervisor", "manager", "senior team",
        "transfer you", "connect you with", "pass this to",
    ],
    "confirmation": [
        "confirmed", "done", "completed", "processed",
        "reference number", "case number", "ticket number",
        "you will receive", "within 24", "within 48",
    ],
    "information_provide": [
        "order status", "tracking number", "shipped", "on its way",
        "delivery date", "estimated arrival", "expected", "dispatch",
        "trk", "order #", "tracking",
    ],
}

# Ordered by priority (first match wins)
_INTENT_PRIORITY = [
    "apology",
    "de_escalation",
    "escalation",
    "confirmation",
    "information_provide",
    "resolution_offer",
    "information_request",
]


def classify_intent(response: str) -> str:
    """
    Classify agent response intent using deterministic keyword matching.
    Returns one of the keys in _INTENT_KEYWORDS, or 'off_topic'.
    O(n·k) — fast, no LLM required.
    """
    lower = response.lower()
    for intent in _INTENT_PRIORITY:
        if any(kw in lower for kw in _INTENT_KEYWORDS[intent]):
            return intent
    return "off_topic"


# ===========================================================================
# RULE-BASED SCORER
# ===========================================================================

POLITE_PHRASES = [
    "sorry", "apologize", "apologies", "thank you", "thanks", "please",
    "happy to help", "glad to assist", "understand your concern",
    "i appreciate", "absolutely", "certainly", "of course", "no problem",
    "my pleasure", "we value", "i assure you",
]

RUDE_WORDS = [
    "shut up", "stupid", "idiot", "useless", "dumb", "ridiculous response",
    "not my problem", "don't care", "whatever", "deal with it",
]


def _rule_based_score(
    response: str,
    conversation_history: List[Dict[str, str]],
) -> float:
    """
    Evaluate quality of agent response via deterministic rules.
    Returns score in [0.0, 1.0].
    """
    score = 0.5  # neutral base

    lower = response.lower()

    if any(phrase in lower for phrase in POLITE_PHRASES):
        score += 0.2

    if any(word in lower for word in RUDE_WORDS):
        score -= 0.3

    if len(response.strip()) < 20:
        score -= 0.2

    # Penalize verbatim repetition of last agent turn
    agent_turns = [m["content"] for m in conversation_history if m["role"] == "assistant"]
    if agent_turns and response.strip() == agent_turns[-1].strip():
        score -= 0.1

    return min(1.0, max(0.0, score))


# ===========================================================================
# STAGE MACHINES (per task)
# ===========================================================================

# Each task defines an ordered list of stage names.
# The stage machine advances when the agent's intent matches the
# "expected_intents" for the current stage.

TASK_STAGES: Dict[str, List[Dict[str, Any]]] = {
    "order_status": [
        {
            "name": "start",
            "expected_intents": {"apology", "information_request", "information_provide"},
            "hint": "Acknowledge the customer's concern and provide order status.",
        },
        {
            "name": "inquiry",
            "expected_intents": {"information_provide", "confirmation"},
            "hint": "Provide the tracking number and expected delivery date.",
        },
        {
            "name": "resolution",
            "expected_intents": {"resolution_offer", "confirmation", "information_provide"},
            "hint": "Confirm resolution and ask if anything else is needed.",
        },
        {
            "name": "closure",
            "expected_intents": {"confirmation", "apology"},
            "hint": "Close the conversation professionally.",
        },
    ],
    "damaged_product": [
        {
            "name": "start",
            "expected_intents": {"apology"},
            "hint": "Immediately apologize for the damaged product.",
        },
        {
            "name": "empathy",
            "expected_intents": {"apology", "de_escalation", "information_request"},
            "hint": "Show empathy and ask for order details to process replacement.",
        },
        {
            "name": "diagnosis",
            "expected_intents": {"information_request", "resolution_offer"},
            "hint": "Confirm the issue and offer a replacement or refund.",
        },
        {
            "name": "resolution",
            "expected_intents": {"resolution_offer", "escalation"},
            "hint": "Commit to replacement/refund with a timeline.",
        },
        {
            "name": "closure",
            "expected_intents": {"confirmation", "apology"},
            "hint": "Provide a reference number and close professionally.",
        },
    ],
    "escalation": [
        {
            "name": "start",
            "expected_intents": {"apology", "de_escalation"},
            "hint": "De-escalate immediately — the customer is very angry.",
        },
        {
            "name": "de_escalation",
            "expected_intents": {"apology", "de_escalation"},
            "hint": "Keep de-escalating and acknowledge every grievance.",
        },
        {
            "name": "acknowledgement",
            "expected_intents": {"resolution_offer", "escalation", "de_escalation"},
            "hint": "Commit to resolution and offer escalation to a manager.",
        },
        {
            "name": "resolution",
            "expected_intents": {"resolution_offer", "escalation", "confirmation"},
            "hint": "Process the refund and connect with supervisor.",
        },
        {
            "name": "closure",
            "expected_intents": {"confirmation"},
            "hint": "Confirm all actions taken and thank the customer.",
        },
    ],
}


# ===========================================================================
# CUSTOMER SCRIPTS (deterministic, stage × mood)
# ===========================================================================

# Structure: CUSTOMER_SCRIPTS[task_name][stage_name][mood] → List[str]
# The environment picks the next message based on stage + mood.
# Multiple variants per combo → cycling through index for determinism.

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
        "success_intents": {"resolution_offer", "information_provide", "confirmation"},
        "failure_conditions": {
            "consecutive_failures": 3,
        },
        "provide_hints": True,  # Easy task — hints enabled
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
        "success_intents": {"resolution_offer"},
        "failure_conditions": {
            "consecutive_failures": 3,
            "no_apology_by_step": 2,  # must apology within 2 steps
        },
        "provide_hints": False,
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
        "success_intents": {"resolution_offer", "escalation"},
        "failure_conditions": {
            "consecutive_failures": 3,
        },
        "provide_hints": False,
    },
}


# ===========================================================================
# LLM JUDGE (called once at episode end)
# ===========================================================================

def _llm_judge_score(
    task_description: str,
    conversation_history: List[Dict[str, str]],
    final_response: str,
) -> float:
    """
    Call LLM via OpenRouter to evaluate the entire episode quality.
    Called ONCE per episode (not per step). Falls back to 0.5 on error.
    """
    try:
        from openai import OpenAI

        llm_base_url = os.getenv("LLM_BASEURL")
        api_key = os.getenv("OPENAI_API_KEY")
        model_name = os.getenv("MODEL_NAME")

        if not all([llm_base_url, api_key, model_name]):
            return 0.5

        client = OpenAI(base_url=llm_base_url, api_key=api_key)

        # Summarize last 6 turns for brevity
        convo_str = "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in conversation_history[-6:]
        )

        judge_prompt = f"""You are an expert customer service quality evaluator.

Task: {task_description}

Conversation Summary (last 6 turns):
{convo_str}

Rate the agent's OVERALL performance in this episode on three criteria (0–10 each):
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

        return 0.5
    except Exception:
        return 0.5


# ===========================================================================
# EPISODE STATE (internal dataclass — not exposed to OpenEnv directly)
# ===========================================================================

@dataclass
class EpisodeState:
    task_name: str
    stages: List[Dict[str, Any]]          # full stage list from TASK_STAGES
    stage_index: int = 0                   # current position in stage list
    customer_mood: str = "neutral"
    issue_status: str = "unresolved"       # unresolved | in_progress | resolved
    steps_taken: int = 0
    max_steps: int = 10
    consecutive_failures: int = 0
    resolved: bool = False
    had_apology: bool = False
    trajectory: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def current_stage(self) -> Dict[str, Any]:
        idx = min(self.stage_index, len(self.stages) - 1)
        return self.stages[idx]

    @property
    def stage_name(self) -> str:
        return self.current_stage["name"]

    @property
    def done(self) -> bool:
        return self.resolved or self.consecutive_failures >= 3 or self.steps_taken >= self.max_steps


# ===========================================================================
# DENSE REWARD COMPUTATION
# ===========================================================================

def _compute_step_reward(
    intent: str,
    rule_score: float,
    stage_advanced: bool,
    episode_state: EpisodeState,
    task: Dict[str, Any],
    is_final_step: bool,
) -> float:
    """
    Compute a dense reward for a single step.
    Combines base score, stage advance bonus, rule quality, and penalties.
    """
    reward = rule_score  # base from rule scorer (0.5 neutral)

    # Stage progression bonus
    if stage_advanced:
        reward += 0.30

    # Empathy / resolution quality bonus (only on relevant intents)
    if intent in {"apology", "de_escalation"}:
        reward += 0.10
    if intent in {"resolution_offer", "confirmation"}:
        reward += 0.10

    # Off-topic penalty
    if intent == "off_topic":
        reward -= 0.20

    # Efficiency bonus at final step (resolved early)
    if is_final_step and episode_state.resolved:
        steps_ratio = episode_state.steps_taken / max(episode_state.max_steps, 1)
        reward += max(0.0, 0.15 * (1.0 - steps_ratio))

    return min(1.0, max(0.0, reward))


# ===========================================================================
# CUSTOMER RESPONSE GENERATION (scripted, deterministic)
# ===========================================================================

def _get_customer_response(
    task_name: str,
    stage_name: str,
    mood: str,
    script_index: int,
) -> str:
    """
    Return a scripted customer response for the given stage + mood.
    Cycles through variants if multiple exist.
    """
    try:
        variants = CUSTOMER_SCRIPTS[task_name][stage_name][mood]
        return variants[script_index % len(variants)]
    except KeyError:
        # Fallback to neutral
        try:
            variants = CUSTOMER_SCRIPTS[task_name][stage_name]["neutral"]
            return variants[script_index % len(variants)]
        except KeyError:
            return "I see. Is there anything else you can do for me?"


def _get_done_message(resolved: bool, mood: str) -> str:
    """Return a terminal customer message."""
    if resolved:
        if mood == "satisfied":
            return "Wonderful! Thank you so much for your help. My issue is fully resolved!"
        elif mood == "neutral":
            return "Thank you for your help. My issue has been resolved."
        else:  # angry but resolved
            return "Fine. I suppose that resolves things. Thank you."
    else:
        return "I'm still not satisfied. I'll need to contact you again or escalate further."


# ===========================================================================
# MOOD EVOLUTION
# ===========================================================================

def _evolve_mood(current_mood: str, intent: str, stage_advanced: bool, rule_score: float) -> str:
    """
    Compute the customer's new mood after the agent's response.

    Rules:
    - Good response (rule_score > 0.65) + stage advanced → mood improves by 1
    - Bad response (rule_score < 0.35) or off_topic → mood worsens by 1
    """
    mood_ladder = ["angry", "neutral", "satisfied"]
    idx = mood_ladder.index(current_mood)

    if (rule_score >= 0.65 and stage_advanced) or intent in {"resolution_offer", "confirmation"}:
        idx = min(idx + 1, 2)
    elif rule_score < 0.35 or intent == "off_topic":
        idx = max(idx - 1, 0)

    return mood_ladder[idx]


# ===========================================================================
# ENVIRONMENT
# ===========================================================================

class CustomerSupportEnvironment(Environment):
    """
    Multi-step, stateful Customer Support Resolution Environment (v2).

    Simulates realistic customer support across 3 difficulty levels with:
      - Stage machine governance (4–5 stages per task)
      - Customer mood evolution (angry → neutral → satisfied)
      - Dense per-step rewards (deterministic)
      - LLM judge at episode end ONLY (1 call per episode)
      - Deterministic grader score in [0.0, 1.0]
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task: Optional[Dict[str, Any]] = None
        self._task_name: str = ""
        self._conversation_history: List[Dict[str, str]] = []
        self._episode: Optional[EpisodeState] = None
        self._script_index: int = 0  # for cycling customer response variants

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
        )

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        # Initial customer message from stage=start + initial mood
        initial_message = _get_customer_response(
            task_name, "start", initial_mood, script_index=0
        )
        self._conversation_history.append({"role": "user", "content": initial_message})

        hints: List[str] = []
        if self._task.get("provide_hints"):
            hints = [stages[0].get("hint", "")]

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
            hints=hints,
            rule_score=0.0,
            llm_score=0.0,
            stage_reward=0.0,
            final_reward=0.0,
            grader_score=0.0,
            done=False,
            reward=0.0,
            task_context=self._task.get("context"),
        )

    # -----------------------------------------------------------------------
    def step(self, action: CustomerSupportAction) -> CustomerSupportObservation:  # type: ignore[override]
        """
        Process the agent's response and advance the episode state.
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
            )

        ep = self._episode
        ep.steps_taken += 1
        self._state.step_count += 1

        agent_response = action.response

        # Record agent turn
        self._conversation_history.append({"role": "assistant", "content": agent_response})

        # ── 1. CLASSIFY INTENT ─────────────────────────────────────────────
        intent = classify_intent(agent_response)

        # ── 2. RULE-BASED SCORE ────────────────────────────────────────────
        rule_score = _rule_based_score(agent_response, self._conversation_history)

        # ── 3. TRACK APOLOGY ──────────────────────────────────────────────
        if intent == "apology":
            ep.had_apology = True

        # ── 4. STAGE MACHINE ADVANCE ───────────────────────────────────────
        current_stage = ep.current_stage
        expected_intents: set = current_stage["expected_intents"]
        stage_advanced = False

        if intent in expected_intents and ep.stage_index < len(ep.stages) - 1:
            ep.stage_index += 1
            stage_advanced = True
            ep.consecutive_failures = 0
            ep.issue_status = "in_progress"
        elif intent == "off_topic" or rule_score < 0.35:
            ep.consecutive_failures += 1
        else:
            # Correct-ish but didn't advance stage — partial credit, reset fail streak
            ep.consecutive_failures = max(0, ep.consecutive_failures - 1)

        # ── 5. RESOLUTION CHECK ────────────────────────────────────────────
        # Resolved if: in closure stage with right intent, OR success intent
        # reached at any point AND issue is in_progress (stage advanced at least once)
        # OR it's a simple task (easy) and a decisive success intent was given.
        is_final_step = ep.done  # evaluate after updating step count
        if ep.stage_name == "closure" and intent in ep.stages[-1]["expected_intents"]:
            ep.resolved = True
            ep.issue_status = "resolved"
        elif (
            intent in self._task.get("success_intents", set())
            and (
                ep.stage_index >= len(ep.stages) - 2  # near final stage
                or self._task["difficulty"] == "easy"  # easy tasks: allow early resolve
            )
        ):
            ep.resolved = True
            ep.issue_status = "resolved"

        is_final_step = ep.done  # re-check after resolution update

        # ── 6. MOOD EVOLUTION ──────────────────────────────────────────────
        ep.customer_mood = _evolve_mood(ep.customer_mood, intent, stage_advanced, rule_score)

        # ── 7. DENSE REWARD ────────────────────────────────────────────────
        step_reward = _compute_step_reward(
            intent=intent,
            rule_score=rule_score,
            stage_advanced=stage_advanced,
            episode_state=ep,
            task=self._task,
            is_final_step=is_final_step,
        )

        stage_reward = 0.30 if stage_advanced else 0.0

        # ── 8. Record step in trajectory FIRST (before grader reads it) ────
        ep.trajectory.append({
            "step": ep.steps_taken,
            "intent": intent,
            "rule_score": rule_score,
            "stage": ep.stage_name,
            "stage_advanced": stage_advanced,
            "mood": ep.customer_mood,
            "reward": step_reward,
        })

        # ── 9. LLM JUDGE (once, at episode end) ───────────────────────────
        llm_score = 0.0
        grader_score = 0.0
        if is_final_step:
            llm_score = _llm_judge_score(
                task_description=self._task["description"],
                conversation_history=self._conversation_history,
                final_response=agent_response,
            )
            # Blend LLM into final step reward
            step_reward = 0.6 * step_reward + 0.4 * llm_score
            # Update trajectory with blended final reward
            ep.trajectory[-1]["reward"] = step_reward

            # ── 10. DETERMINISTIC GRADER (reads trajectory already populated) ─
            grader_score = grade_episode(
                trajectory=ep.trajectory,
                final_stage=ep.stage_name,
                final_mood=ep.customer_mood,
                resolved=ep.resolved,
                steps_taken=ep.steps_taken,
                max_steps=ep.max_steps,
            )

        # ── 10. NEXT CUSTOMER MESSAGE ──────────────────────────────────────
        self._script_index += 1
        if is_final_step:
            next_message = _get_done_message(ep.resolved, ep.customer_mood)
        else:
            # Use the NEW stage after potential advance
            next_stage_name = ep.stage_name
            next_message = _get_customer_response(
                self._task_name, next_stage_name, ep.customer_mood, self._script_index
            )

        if not is_final_step:
            self._conversation_history.append({"role": "user", "content": next_message})

        # ── 11. HINTS (easy task only) ─────────────────────────────────────
        hints: List[str] = []
        if self._task.get("provide_hints") and not is_final_step:
            next_stage_info = ep.current_stage
            if "hint" in next_stage_info:
                hints = [next_stage_info["hint"]]

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
            intent_detected=intent,
            hints=hints,
            rule_score=rule_score,
            llm_score=llm_score,
            stage_reward=stage_reward,
            final_reward=step_reward,
            grader_score=grader_score,
            done=is_final_step,
            reward=step_reward,
            task_context=self._task.get("context"),
        )

    # -----------------------------------------------------------------------
    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state
