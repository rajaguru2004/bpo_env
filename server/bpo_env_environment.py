"""
Customer Support Environment Implementation.

Simulates real-world customer support conversations. The agent (LLM) acts as
a customer support executive, resolving customer issues across 3 difficulty levels.
Uses a hybrid reward system: rule-based scoring + LLM judge scoring.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional
from uuid import uuid4

# Load environment variables from .env if it exists
# This handles cases where the server is started without sourcing .env
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
if os.path.exists(env_path):
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        # Simple manual fallback for .env loading
        with open(env_path) as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    name, value = line.split("=", 1)
                    os.environ[name.strip()] = value.strip().strip('"').strip("'")

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

import sys
import os

# Ensure the root directory is in sys.path for absolute imports
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

try:
    from models import CustomerSupportAction, CustomerSupportObservation
except ImportError:
    from ..models import CustomerSupportAction, CustomerSupportObservation


# ---------------------------------------------------------------------------
# Task Definitions
# ---------------------------------------------------------------------------

TASKS: Dict[str, Dict[str, Any]] = {
    "order_status": {
        "difficulty": "easy",
        "max_steps": 5,
        "description": "Customer wants to know the status of their order #12345.",
        "initial_message": (
            "Hi, I placed an order (#12345) three days ago and I haven't received "
            "any shipping confirmation yet. Can you please check the status?"
        ),
        "context": {
            "order_id": "12345",
            "order_date": "2026-03-29",
            "status": "Shipped",
            "tracking_number": "TRK987654321",
            "expected_delivery": "2026-04-03",
        },
        # Resolution is detected when the agent mentions the order/tracking info
        "resolution_keywords": [
            "tracking", "shipped", "delivery", "order status",
            "trk", "your order", "on its way", "dispatch"
        ],
        "follow_up_messages": [
            "Can you tell me the tracking number too?",
            "When will it be delivered approximately?",
            "Thank you, that's all I needed.",
        ],
    },
    "damaged_product": {
        "difficulty": "medium",
        "max_steps": 8,
        "description": "Customer received a damaged product and wants a replacement.",
        "initial_message": (
            "I received my package today but the product inside was completely damaged. "
            "The box was crushed and the item is broken. This is unacceptable! "
            "I need a replacement or a refund immediately."
        ),
        "context": {
            "order_id": "98765",
            "product": "Bluetooth Speaker",
            "issue": "Damaged on arrival",
        },
        "resolution_keywords": [
            "replacement", "replace", "refund", "apologize", "sorry",
            "send a new", "new unit", "compensation", "arrange", "ship out"
        ],
        "follow_up_messages": [
            "How long will the replacement take to arrive?",
            "Do I need to return the damaged item?",
            "Can I get a confirmation number for this replacement?",
            "Alright, thank you for resolving this.",
        ],
    },
    "escalation": {
        "difficulty": "hard",
        "max_steps": 12,
        "description": (
            "Angry customer demanding a refund and to speak with a manager. "
            "Multi-turn escalation scenario."
        ),
        "initial_message": (
            "This is absolutely ridiculous! I've been waiting for 2 weeks and my "
            "order STILL hasn't arrived. I've called 3 times already and nobody "
            "helps me! I want a FULL REFUND and I want to speak to your manager RIGHT NOW!"
        ),
        "context": {
            "order_id": "55501",
            "product": "Laptop Stand",
            "days_delayed": 14,
            "prior_contacts": 3,
        },
        "resolution_keywords": [
            "escalate", "manager", "supervisor", "refund", "full refund",
            "sincerely apologize", "deeply sorry", "transfer you",
            "connect you", "compensation"
        ],
        "follow_up_messages": [
            "You're just reading from a script! I want real answers!",
            "How long will the refund take?",
            "Can I speak to a manager directly?",
            "Nobody has taken responsibility for this mess.",
            "Fine, if you can confirm the refund and escalate, I'll wait.",
            "Okay, I appreciate you handling this. Thank you.",
        ],
    },
}


# ---------------------------------------------------------------------------
# Polite / rude keyword lists for rule-based scoring
# ---------------------------------------------------------------------------

POLITE_PHRASES = [
    "sorry", "apologize", "apologies", "thank you", "thanks", "please",
    "happy to help", "glad to assist", "understand your concern",
    "i appreciate", "absolutely", "certainly", "of course", "no problem",
    "my pleasure", "we value", "i assure you",
]

RUDE_WORDS = [
    "shut up", "stupid", "idiot", "useless", "dumb", "ridiculous response",
    "not my problem", "don't care", "whatever",
]


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------

def _llm_judge_score(
    task_description: str,
    conversation_history: List[Dict[str, str]],
    agent_response: str,
) -> float:
    """
    Call LLM via OpenRouter API to evaluate the agent's response.
    Returns a normalized score 0.0–1.0. Falls back to 0.5 on error.
    """
    try:
        from openai import OpenAI

        llm_base_url = os.getenv("LLM_BASEURL")
        api_key = os.getenv("OPENAI_API_KEY")
        model_name = os.getenv("MODEL_NAME")

        if not all([llm_base_url, api_key, model_name]):
            return 0.5  # fallback if env vars not set

        client = OpenAI(base_url=llm_base_url, api_key=api_key)

        # Build concise conversation string for the judge
        convo_str = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in conversation_history[-6:]  # last 6 turns for brevity
        )

        judge_prompt = f"""You are an expert customer service quality evaluator.

Task: {task_description}

Recent Conversation:
{convo_str}

Latest Agent Response:
"{agent_response}"

Evaluate the agent's response on these criteria (score each 0-10):
1. Helpfulness: Did the agent actually help with the customer's issue?
2. Correctness: Was the information/solution provided accurate and appropriate?
3. Tone: Was the agent professional, empathetic, and polite?

Respond ONLY with valid JSON in this exact format:
{{"helpfulness": <0-10>, "correctness": <0-10>, "tone": <0-10>}}"""

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a customer service quality evaluator. Always respond with valid JSON only.",
                },
                {"role": "user", "content": judge_prompt},
            ],
            max_tokens=100,
            temperature=0.1,
        )

        content = response.choices[0].message.content.strip()

        # Extract JSON from response
        json_match = re.search(r"\{[^}]+\}", content)
        if json_match:
            scores = json.loads(json_match.group())
            helpfulness = float(scores.get("helpfulness", 5))
            correctness = float(scores.get("correctness", 5))
            tone = float(scores.get("tone", 5))
            avg = (helpfulness + correctness + tone) / 3.0
            return min(1.0, max(0.0, avg / 10.0))

        return 0.5  # fallback if parsing fails

    except Exception:
        return 0.5  # silent fallback on any error


# ---------------------------------------------------------------------------
# Rule-Based Scorer
# ---------------------------------------------------------------------------

def _rule_based_score(
    response: str,
    conversation_history: List[Dict[str, str]],
) -> float:
    """
    Evaluate the agent's response using deterministic rules.
    Returns a score 0.0–1.0.
    """
    score = 0.5  # neutral base

    lower_resp = response.lower()

    # +0.2 for polite/empathetic language
    if any(phrase in lower_resp for phrase in POLITE_PHRASES):
        score += 0.2

    # -0.3 for rude language
    if any(word in lower_resp for word in RUDE_WORDS):
        score -= 0.3

    # -0.2 for very short responses (unhelpful)
    if len(response.strip()) < 20:
        score -= 0.2

    # -0.1 for verbatim repetition of last agent turn
    agent_turns = [
        msg["content"] for msg in conversation_history if msg["role"] == "assistant"
    ]
    if agent_turns and response.strip() == agent_turns[-1].strip():
        score -= 0.1

    return min(1.0, max(0.0, score))


# ---------------------------------------------------------------------------
# Resolution Check
# ---------------------------------------------------------------------------

def _check_resolution(response: str, resolution_keywords: List[str]) -> bool:
    """Check if the agent's response contains resolution signals."""
    lower_resp = response.lower()
    return any(keyword.lower() in lower_resp for keyword in resolution_keywords)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class CustomerSupportEnvironment(Environment):
    """
    Multi-turn Customer Support Resolution Environment.

    Simulates real-world customer support conversations across 3 difficulty levels:
    - easy (order_status): Simple order tracking query
    - medium (damaged_product): Product complaint & replacement
    - hard (escalation): Angry customer demanding refund + manager

    Reward is a hybrid combination of rule-based scoring and LLM judge scoring.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task: Optional[Dict[str, Any]] = None
        self._task_name: str = ""
        self._conversation_history: List[Dict[str, str]] = []
        self._current_message: str = ""
        self._follow_up_index: int = 0
        self._is_resolved: bool = False
        self._done: bool = False

    def reset(
        self,
        task_name: Optional[str] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> CustomerSupportObservation:
        """
        Reset the environment to start a new episode.

        Args:
            task_name: One of 'order_status', 'damaged_product', 'escalation'.
                      Defaults to 'order_status' if not provided.
            episode_id: Optional episode ID override.

        Returns:
            CustomerSupportObservation with the first customer message.
        """
        # Default to easy task if none specified
        if task_name is None or task_name not in TASKS:
            task_name = "order_status"

        self._task_name = task_name
        self._task = TASKS[task_name]
        self._conversation_history = []
        self._follow_up_index = 0
        self._is_resolved = False
        self._done = False

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        # Set initial customer message
        self._current_message = self._task["initial_message"]

        # Add to history
        self._conversation_history.append(
            {"role": "user", "content": self._current_message}
        )

        return CustomerSupportObservation(
            customer_message=self._current_message,
            conversation_history=list(self._conversation_history),
            task_name=self._task_name,
            task_difficulty=self._task["difficulty"],
            step=0,
            max_steps=self._task["max_steps"],
            is_resolved=False,
            rule_score=0.0,
            llm_score=0.0,
            final_reward=0.0,
            done=False,
            reward=0.0,
            task_context=self._task.get("context"),
        )

    def step(self, action: CustomerSupportAction) -> CustomerSupportObservation:  # type: ignore[override]
        """
        Process the agent's response and advance the episode.

        Args:
            action: CustomerSupportAction with the agent's response text.

        Returns:
            CustomerSupportObservation with next customer message and reward scores.
        """
        if self._task is None:
            # Environment not reset yet — return empty terminal obs
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

        self._state.step_count += 1
        agent_response = action.response

        # Record agent response in history
        self._conversation_history.append(
            {"role": "assistant", "content": agent_response}
        )

        # --- REWARD COMPUTATION ---
        rule_score = _rule_based_score(agent_response, self._conversation_history)

        llm_score = _llm_judge_score(
            task_description=self._task["description"],
            conversation_history=self._conversation_history,
            agent_response=agent_response,
        )

        final_reward = 0.5 * rule_score + 0.5 * llm_score

        # --- RESOLUTION CHECK ---
        resolved = _check_resolution(
            agent_response, self._task["resolution_keywords"]
        )
        if resolved:
            self._is_resolved = True

        # --- TERMINATION CHECK ---
        max_steps_reached = self._state.step_count >= self._task["max_steps"]
        self._done = self._is_resolved or max_steps_reached

        # --- NEXT CUSTOMER MESSAGE ---
        follow_ups: List[str] = self._task.get("follow_up_messages", [])

        if self._done:
            next_message = (
                "Thank you for your help! My issue has been resolved."
                if self._is_resolved
                else "I'm still not satisfied. I'll have to contact you again."
            )
        elif self._follow_up_index < len(follow_ups):
            next_message = follow_ups[self._follow_up_index]
            self._follow_up_index += 1
        else:
            next_message = "I see. Is there anything else you can do for me?"

        if not self._done:
            self._conversation_history.append(
                {"role": "user", "content": next_message}
            )
        self._current_message = next_message

        return CustomerSupportObservation(
            customer_message=next_message,
            conversation_history=list(self._conversation_history),
            task_name=self._task_name,
            task_difficulty=self._task["difficulty"],
            step=self._state.step_count,
            max_steps=self._task["max_steps"],
            is_resolved=self._is_resolved,
            rule_score=rule_score,
            llm_score=llm_score,
            final_reward=final_reward,
            done=self._done,
            reward=final_reward,
            task_context=self._task.get("context"),
        )

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state
