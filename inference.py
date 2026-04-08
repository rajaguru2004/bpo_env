"""
inference.py — BPO Customer Support Environment Inference Script

Runs all 3 tasks automatically using an LLM agent via the hackathon-injected API.
The agent acts as a customer support executive responding to customer queries.

Usage:
    python inference.py

Environment Variables (Hackathon Validator injects these):
    API_KEY          — LLM proxy API key  (primary; fallback: HF_TOKEN)
    API_BASE_URL     — LLM proxy base URL (default: https://router.huggingface.co/v1)
    MODEL_NAME       — LLM model to use
    IMAGE_NAME       — Docker image name OR http(s):// URL for the environment server
                       (primary; fallback: LOCAL_IMAGE_NAME)

Local Dev Extras (.env):
    HF_TOKEN         — Fallback if API_KEY not set
    SERVER_URL       — Used when neither IMAGE_NAME nor LOCAL_IMAGE_NAME is set
    LOCAL_IMAGE_NAME — Fallback docker image for local dev
"""

import os
import subprocess
import sys
import time
import textwrap
from typing import Any, Dict, List, Optional, Set

# ---------------------------------------------------------------------------
# Path Initialization (Must be before internal imports)
# ---------------------------------------------------------------------------
_project_root = os.path.dirname(os.path.abspath(__file__))
_parent_root = os.path.dirname(_project_root)

# Standardize sys.path for both local execution and IDE linter (Pyrefly)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _parent_root not in sys.path:
    sys.path.insert(0, _parent_root)

# ---------------------------------------------------------------------------
# Load .env for local development
# ---------------------------------------------------------------------------
_env_path = os.path.join(_project_root, ".env")
if os.path.exists(_env_path):
    try:
        import importlib
        _dotenv = importlib.import_module("dotenv")
        _dotenv.load_dotenv(_env_path)
    except (ImportError, ModuleNotFoundError):
        # Minimal manual parsing if dotenv is missing or linter is blind
        with open(_env_path) as _f:
            for _line in _f:
                if "=" in _line and not _line.startswith("#"):
                    _name, _value = _line.split("=", 1)
                    os.environ.setdefault(_name.strip(), _value.strip().strip('"').strip("'"))

# ---------------------------------------------------------------------------
# Configuration — Validator-injected variables take priority
# ---------------------------------------------------------------------------

# API credentials: validator injects API_KEY; local dev uses HF_TOKEN
API_KEY = (
    os.getenv("API_KEY")
    or os.getenv("HF_TOKEN")
    or ""
)

# LLM router base URL
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")

# Model to use for inference
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# Docker image / server URL: validator injects IMAGE_NAME; local dev uses LOCAL_IMAGE_NAME
IMAGE_NAME = (
    os.getenv("IMAGE_NAME")
    or os.getenv("LOCAL_IMAGE_NAME")
    or ""
)

# Fallback local server URL (used when IMAGE_NAME is also absent)
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")

# Benchmark identifier (matches openenv.yaml name)
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", os.getenv("BENCHMARK", "bpo_env"))

# Task selection:
# - Hackathon evaluator injects MY_ENV_V4_TASK as a REAL process env var → run that one task
# - Not set → run all 3 tasks (local verification mode)
# NOTE: We explicitly ignore .env for task selection to prevent accidental overrides.
_TASK_FROM_EVALUATOR = os.environ.get("MY_ENV_V4_TASK") or os.environ.get("TASK_NAME")

# Success threshold for score-based resolution
SUCCESS_SCORE_THRESHOLD = 0.5

# Default max steps if observation doesn't supply it
MAX_STEPS_DEFAULT = 10

# All task IDs (used in multi-task mode)
TASKS_TO_RUN = ["task_easy", "task_medium", "task_hard"]

# ---------------------------------------------------------------------------
# OpenAI-compatible client (resolves after env vars are loaded)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# OpenAI-compatible client (resolves after env vars are loaded)
# ---------------------------------------------------------------------------
try:
    import importlib
    _openai_mod = importlib.import_module("openai")
    OpenAI = _openai_mod.OpenAI
except (ImportError, ModuleNotFoundError):
    # Fallback if openai is completely missing
    OpenAI = None # type: ignore

if OpenAI:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
else:
    client = None

# ---------------------------------------------------------------------------
# Robustness module imports (Internal Only - agent side)
# ---------------------------------------------------------------------------
try:
    # 1. Try IDE-friendly prefixed imports first
    from bpo_env.agent_logic.response_validator import ResponseValidator, ResponseValidatorState
    from bpo_env.agent_logic.stage_policy_enforcer import StagePolicyEnforcer
    from bpo_env.agent_logic.anti_stall_engine import AntiStallEngine, AntiStallState
    from bpo_env.agent_logic.episode_memory import EpisodeMemory
    from bpo_env.agent_logic.repeat_intent_detector import RepeatIntentDetector
    from bpo_env.agent_logic.stage_sequence_guard import StageSequenceGuard
    from bpo_env.agent_logic.intents import extract_mood, extract_intents, get_bridge_intents
    from bpo_env.agent_logic.immediate_recovery_policy import ImmediateRecoveryPolicy
    from bpo_env.agent_logic.closure_enforcer import ClosureEnforcer
    from bpo_env.agent_logic.mood_adaptive_policy import MoodAdaptivePolicy
except ImportError:
    try:
        # 2. Dynamic fallback (hides from IDE linter but works at runtime)
        import importlib
        _validator_mod = importlib.import_module("agent_logic.response_validator")
        ResponseValidator = _validator_mod.ResponseValidator
        ResponseValidatorState = _validator_mod.ResponseValidatorState
        
        _enforcer_mod = importlib.import_module("agent_logic.stage_policy_enforcer")
        StagePolicyEnforcer = _enforcer_mod.StagePolicyEnforcer
        
        _stall_mod = importlib.import_module("agent_logic.anti_stall_engine")
        AntiStallEngine = _stall_mod.AntiStallEngine
        AntiStallState = _stall_mod.AntiStallState
        
        _memory_mod = importlib.import_module("agent_logic.episode_memory")
        EpisodeMemory = _memory_mod.EpisodeMemory
        
        _repeat_mod = importlib.import_module("agent_logic.repeat_intent_detector")
        RepeatIntentDetector = _repeat_mod.RepeatIntentDetector
        
        _guard_mod = importlib.import_module("agent_logic.stage_sequence_guard")
        StageSequenceGuard = _guard_mod.StageSequenceGuard
        
        _intents_mod = importlib.import_module("agent_logic.intents")
        extract_mood = _intents_mod.extract_mood
        extract_intents = _intents_mod.extract_intents
        get_bridge_intents = _intents_mod.get_bridge_intents
        
        _recovery_mod = importlib.import_module("agent_logic.immediate_recovery_policy")
        ImmediateRecoveryPolicy = _recovery_mod.ImmediateRecoveryPolicy
        
        _closure_mod = importlib.import_module("agent_logic.closure_enforcer")
        ClosureEnforcer = _closure_mod.ClosureEnforcer
        
        _mood_mod = importlib.import_module("agent_logic.mood_adaptive_policy")
        MoodAdaptivePolicy = _mood_mod.MoodAdaptivePolicy
        
    except (ImportError, ModuleNotFoundError):
        # 3. Final default fallback
        ResponseValidator = ResponseValidatorState = StagePolicyEnforcer = None  # type: ignore
        AntiStallEngine = AntiStallState = EpisodeMemory = None # type: ignore
        RepeatIntentDetector = StageSequenceGuard = extract_mood = None # type: ignore
        extract_intents = get_bridge_intents = ImmediateRecoveryPolicy = None # type: ignore
        ClosureEnforcer = MoodAdaptivePolicy = None # type: ignore

# Phrase diversity banks — Task 3.C
_PHRASE_POOLS = {
    "apology": [
        "I completely understand your frustration and I sincerely apologize for this experience.",
        "I am truly sorry for the inconvenience this has caused you.",
        "I deeply apologize for the trouble you've experienced with us.",
        "Your experience is unacceptable and I sincerely apologize on behalf of our team.",
        "I hear you, and I want to fix this for you right now."
    ],
    "resolution": [
        "I will process this resolution for you immediately to resolve the issue.",
        "Let's get this settled right away for you.",
        "I am taking care of this now so you can have peace of mind.",
        "To make this right, I'll process the request instantly.",
    ],
    "closure": [
        "Is there anything else I can assist with today?",
        "I'm here to help if you have further questions.",
        "Please let me know if there's anything else I can do for you.",
    ]
}
_empathy_rotation_idx = 0

# ---------------------------------------------------------------------------
# Agent system prompt
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a professional and empathetic customer support executive at a leading
    e-commerce company.

    Your responsibilities:
    - Listen carefully to customer concerns
    - Provide accurate, helpful, and actionable information
    - Maintain a polite, empathetic, and professional tone at all times
    - Resolve issues efficiently within the conversation
    - Offer appropriate solutions (replacements, refunds, escalation when needed)

    Always respond in 2-4 sentences. Be concrete and solution-focused.\
""")

# ---------------------------------------------------------------------------
# Module-level EpisodeMemory singleton (shared across tasks in one session)
# ---------------------------------------------------------------------------
_episode_memory: Optional[Any] = EpisodeMemory() if EpisodeMemory else None


# ---------------------------------------------------------------------------
# Quick intent extractor (lightweight bridge for robustness modules)
# Does NOT call the environment — used only before submitting to env.
# ---------------------------------------------------------------------------
def _get_draft_intents(text: str, task_name: str) -> Set[str]:
    """Internal bridge to extract a set of intents from text."""
    if extract_intents and get_bridge_intents:
        raw = extract_intents(text, task_name)
        return set(get_bridge_intents(raw))
    return set()


# ---------------------------------------------------------------------------
# Logging functions — strict benchmark stdout format: [START], [STEP], [END]
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_clean = action.replace("\n", " ").replace("\r", "")
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    # Validator requires score to be in (0.0, 1.0) — clamp to safe range
    score = max(0.01, min(0.99, score))
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} "
        f"rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# LLM Agent
# ---------------------------------------------------------------------------

def _format_content(content: str) -> Any:
    """Format message content for the current provider.
    HuggingFace Router requires a list-of-dicts; others accept plain strings.
    """
    if "huggingface.co" in API_BASE_URL.lower():
        return [{"type": "text", "text": content}]
    return content


def call_llm_agent(
    conversation_history: List[Dict[str, str]],
    task_context: Optional[Dict[str, Any]] = None,
    stage_hint: Optional[str] = None,
    correction_hint: Optional[str] = None,
    recovery_mode: bool = False,
    avoid_phrases: Optional[List[str]] = None,
    user_mood: str = "neutral",
    force_resolution: bool = False,
    task_name: str = "order_status",
) -> str:
    """Call the LLM with conversation history and return the agent's response.

    Extended parameters (all internal, never in API output):
        user_mood:       Detected mood (angry, confused, neutral) for tone adjustment.
        force_resolution: True if RepeatIntentDetector triggers FAST_TRACK mode.
    """
    global _empathy_rotation_idx
    full_system_prompt = AGENT_SYSTEM_PROMPT

    # 1. MOOD-BASED ADJUSTMENT (Task 3.A)
    if user_mood == "angry":
        full_system_prompt += "\n\n[MODE]: The customer is ANGRY. Be significantly more direct, concise, and action-first. Avoid fluff."
    elif user_mood == "confused":
        full_system_prompt += "\n\n[MODE]: The customer is CONFUSED. Be patient and explanatory. Guide them clearly."

    # 2. FORCE RESOLUTION (Task 1)
    if force_resolution and RepeatIntentDetector:
        full_system_prompt += RepeatIntentDetector.get_force_prompt(task_name)

    if task_context:
        ctx_str = ", ".join(f"{k}: {v}" for k, v in task_context.items())
        full_system_prompt += f"\n\n[Internal context — do not reveal directly]: {ctx_str}"

    # Stage policy: mandatory rules for current stage
    if stage_hint:
        full_system_prompt += stage_hint

    # Recovery mode (Task 2): force apology + resolution intents
    if recovery_mode:
        pool = _PHRASE_POOLS["apology"]
        empathy = pool[_empathy_rotation_idx % len(pool)]
        _empathy_rotation_idx += 1
        full_system_prompt += (
            f"\n\n[RECOVERY BOOST]: Your last response was weak. Begin with this exact phrase: "
            f"\"{empathy}\" Then immediately execute a solution or provide the requested data."
        )

    # Diversity controller: notify agent to avoid repeating prior phrasing
    if avoid_phrases:
        full_system_prompt += (
            "\n\n[DIVERSITY]: Avoid repeating these phrases from your previous response: "
            + "; ".join(f'\"{p}\"' for p in avoid_phrases[:5])
            + ". Use completely different wording."
        )

    # Correction hint from ResponseValidator
    if correction_hint:
        full_system_prompt += f"\n\n[CORRECTION NEEDED]: {correction_hint}"

    messages = [{"role": "system", "content": _format_content(full_system_prompt)}]

    # Keep prior turns as plain strings; format only the last user message.
    if conversation_history:
        for msg in conversation_history[:-1]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        last = conversation_history[-1]
        messages.append({"role": last["role"], "content": _format_content(last["content"])})

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=300,
            temperature=0.7,
        )
        content = resp.choices[0].message.content.strip()
        return content if content else _fallback_response(conversation_history)
    except Exception as exc:
        print(f"[WARN] LLM API error: {exc}", file=sys.stderr, flush=True)
        return _fallback_response(conversation_history)


def _fallback_response(conversation_history: List[Dict[str, str]]) -> str:
    last_user = ""
    for msg in reversed(conversation_history):
        if msg["role"] == "user":
            last_user = msg["content"]
            break

    lower = last_user.lower()
    if any(w in lower for w in ["refund", "money", "charge"]):
        return (
            "I sincerely apologize for the inconvenience. I will process your refund "
            "immediately and escalate this to our senior team. You will receive a "
            "confirmation within 24 hours."
        )
    if any(w in lower for w in ["damage", "broken", "defect"]):
        return (
            "I'm truly sorry to hear your product arrived damaged. I will arrange a "
            "replacement to be shipped out right away at no additional cost to you. "
            "Please allow 3-5 business days for delivery."
        )
    return (
        "Thank you for reaching out! Your order is currently being processed and "
        "you should receive an update within 24 hours. Please let me know if there "
        "is anything else I can assist with."
    )

# ---------------------------------------------------------------------------
# Server / Environment connection helpers
# ---------------------------------------------------------------------------

def wait_for_server(url: str, timeout: int = 30) -> bool:
    """Poll /health until the server responds or timeout is reached."""
    import urllib.request, urllib.error
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(f"{url}/health", timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def _resolve_server_url() -> str:
    """
    Determine the environment server URL to use.

    Priority:
      1. IMAGE_NAME is an http(s):// URL → use it directly (HF Space or remote server)
      2. IMAGE_NAME is a Docker image tag  → start container via `docker run`
      3. No IMAGE_NAME → check SERVER_URL (already running local server)
      4. Final fallback: start uvicorn locally from source.
    """
    if IMAGE_NAME:
        if IMAGE_NAME.startswith(("http://", "https://")):
            # Remote server / HF Space URL — use directly
            return IMAGE_NAME
        else:
            # Docker image tag — start container directly via docker run
            subprocess.Popen(
                [
                    "docker", "run", "--rm", "-d",
                    "-p", "8000:8000",
                    "--name", "openenv-bpo-inference",
                    IMAGE_NAME,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if not wait_for_server(SERVER_URL, timeout=30):
                print("[ERROR] Docker container failed to start within 30s.", file=sys.stderr, flush=True)
                sys.exit(1)
            return SERVER_URL

    # Try the configured SERVER_URL first (already running server)
    if wait_for_server(SERVER_URL, timeout=5):
        return SERVER_URL

    # Last resort: spawn uvicorn locally from source
    subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "server.app:app", "--host", "0.0.0.0", "--port", "8000",
        ],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not wait_for_server(SERVER_URL, timeout=25):
        print("[ERROR] Could not start server. Ensure uvicorn is installed.", file=sys.stderr, flush=True)
        sys.exit(1)
    return SERVER_URL


# ---------------------------------------------------------------------------
# Task runner dependencies
# ---------------------------------------------------------------------------
try:
    from bpo_env.client import CustomerSupportEnv
    from bpo_env.models import CustomerSupportAction
except ImportError:
    try:
        import importlib
        _c_mod = importlib.import_module("client")
        _m_mod = importlib.import_module("models")
        CustomerSupportEnv = _c_mod.CustomerSupportEnv
        CustomerSupportAction = _m_mod.CustomerSupportAction
    except (ImportError, ModuleNotFoundError):
        CustomerSupportEnv = CustomerSupportAction = None # type: ignore

def run_task(task_name: str, server_url: str) -> Dict[str, Any]:
    """Run a single task episode and return a summary dict."""
    results: Dict[str, Any] = {
        "task_name": task_name,
        "steps": 0,
        "total_reward": 0.0,
        "rewards": [],
        "rule_scores": [],
        "resolved": False,
        "success": False,
        "score": 0.0,
        "avg_rule_score": 0.0,
        "grader_score": 0.0,
    }

    env_client = CustomerSupportEnv(base_url=server_url).sync()
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    # ── Robustness module instances (per-episode, never in API output) ─────────
    validator_state = ResponseValidatorState() if ResponseValidatorState else None
    stall_state = AntiStallState() if AntiStallState else None
    memory = _episode_memory  # shared across runs within a session
    
    # New detectors (Task 1, 5)
    user_input_history: List[str] = []
    history_intents: List[Set[str]] = []

    try:
        with env_client as env:
            # --- RESET ---
            try:
                result = env.reset(task_name=task_name)
                obs = result.observation
                done = result.done
            except Exception as exc:
                print(f"[WARN] Reset failed: {exc}", file=sys.stderr, flush=True)
                log_end(success=False, steps=0, score=0.01, rewards=[])
                return results

            conversation_history = obs.conversation_history
            task_context = obs.task_context
            max_steps = obs.max_steps or MAX_STEPS_DEFAULT
            current_stage = getattr(obs, "conversation_stage", "start")
            internal_task_name = getattr(obs, "task_name", task_name)

            rewards: List[float] = []
            step = 0
            last_agent_response = ""
            last_reward = 1.0
            
            # Anti-Loop (Task 6)
            intent_counts: Dict[str, int] = {}

            # --- STEP LOOP ---
            while not done and step < max_steps:
                step += 1

                # 1. Stage policy hint (StagePolicyEnforcer)
                stage_hint = ""
                if StagePolicyEnforcer:
                    stage_hint = StagePolicyEnforcer.build_policy_prompt(
                        internal_task_name, current_stage
                    )

                # 2. Anti-stall hint (AntiStallEngine)
                stall_hint = ""
                if AntiStallEngine and stall_state:
                    intents_so_far: Set[str] = set()
                    _hint = AntiStallEngine.get_unstick_hint(
                        stall_state, intents_so_far, internal_task_name, False
                    )
                    if _hint:
                        stall_hint = f"\n\n{_hint}"

                # 3. EpisodeMemory few-shot hint
                memory_hint = ""
                if memory:
                    _mhint = memory.build_few_shot_hint(internal_task_name, current_stage)
                    if _mhint:
                        memory_hint = _mhint

                # 4. Diversity controller: extract key phrases from last response
                avoid_phrases: List[str] = []
                if last_agent_response:
                    # Extract bigrams/trigrams to suppress repetition
                    words = last_agent_response.split()
                    avoid_phrases = [
                        " ".join(words[i:i+3])
                        for i in range(0, min(len(words) - 2, 15), 3)
                    ]

                # 5. Pipeline Phase: Input Analysis & Policy Hints (Task 6)
                current_user_msg = conversation_history[-1]["content"] if conversation_history else ""
                
                # 5a. Intent & Mood Detection
                user_mood = extract_mood(current_user_msg) if extract_mood else "neutral"
                current_user_intents = _get_draft_intents(current_user_msg, internal_task_name)
                
                # 5b. Repeat Intent Detection (FORCE_FINAL_RESOLUTION)
                force_res = False
                if RepeatIntentDetector:
                    force_res = RepeatIntentDetector.should_force_resolution(
                        current_user_msg, user_input_history, current_user_intents, 
                        full_intent_history=history_intents
                    )
                
                # 5c. Immediate Recovery Policy
                recovery_hint = ""
                if ImmediateRecoveryPolicy:
                    recovery_hint = ImmediateRecoveryPolicy.get_recovery_hint(internal_task_name, last_reward) or ""
                
                # 5d. Mood Adaptive Policy
                mood_hint = ""
                if MoodAdaptivePolicy:
                    mood_hint = MoodAdaptivePolicy.get_mood_hint(user_mood) or ""

                # 5e. Closure Enforcer Hint (Pre-emptive)
                closure_hint = ""
                if ClosureEnforcer:
                    # Look for resolution intents in previous turns or agent's mind
                    last_agent_intents = history_intents[-1] if history_intents else set()
                    closure_hint = ClosureEnforcer.get_closure_hint(current_stage, last_agent_intents) or ""

                # Combined correction hint from Stage + Stall + Memory + Policies
                combined_hint = (stage_hint + stall_hint + memory_hint + recovery_hint + mood_hint + closure_hint) or None

                # Update user history
                user_input_history.append(current_user_msg)

                # 6. Generate agent response (Response Generator)
                agent_response = call_llm_agent(
                    conversation_history,
                    task_context,
                    stage_hint=combined_hint,
                    correction_hint=None,
                    recovery_mode=False, # Replaced by ImmediateRecoveryPolicy hint
                    avoid_phrases=avoid_phrases if avoid_phrases else None,
                    user_mood=user_mood,
                    force_resolution=force_res,
                    task_name=internal_task_name,
                )

                # 7. Post-Processing / Finalization Pipeline
                if ResponseValidator and validator_state:
                    draft_intents = _get_draft_intents(agent_response, internal_task_name)
                    
                    validation = ResponseValidator.validate(
                        draft_response=agent_response,
                        intents=draft_intents,
                        task_name=internal_task_name,
                        stage_name=current_stage,
                        state=validator_state,
                    )
                    if not validation.is_valid:
                        agent_response = call_llm_agent(
                            conversation_history,
                            task_context,
                            stage_hint=combined_hint,
                            correction_hint=validation.correction_hint,
                            user_mood=user_mood,
                            force_resolution=force_res,
                            task_name=internal_task_name,
                        )

                # 7a. Closure Enforcer (Final check / Injection)
                if ClosureEnforcer:
                    final_intents = _get_draft_intents(agent_response, internal_task_name)
                    agent_response = ClosureEnforcer.ensure_closure_phrase(agent_response, final_intents)
                
                # 7b. Stage Sequence Guard
                if StageSequenceGuard:
                    current_intents = _get_draft_intents(agent_response, internal_task_name)
                    guard_res = StageSequenceGuard.check_sequence(
                        internal_task_name, current_stage, current_intents, history_intents
                    )
                    if not guard_res.is_ordered:
                        injection = StageSequenceGuard.get_repair_injection(guard_res.missing_intent)
                        agent_response = f"{injection} {agent_response}"
                
                # 7b. Anti-Loop Hard Stop (Task 6)
                current_intents = _get_draft_intents(agent_response, internal_task_name)
                for intent in current_intents:
                    intent_counts[intent] = intent_counts.get(intent, 0) + 1
                    if intent_counts[intent] >= 3 and intent not in {"greeting", "confirmation"}:
                        # Force intent diversificaton - re-draft one last time
                        agent_response = call_llm_agent(
                            conversation_history,
                            task_context,
                            stage_hint=combined_hint + f"\n\n[STRICT]: Avoid using the '{intent}' intent again. Transition now.",
                            user_mood=user_mood,
                        )
                        break

                # Send step to environment
                try:
                    if CustomerSupportAction is None:
                        raise ImportError("CustomerSupportAction not imported.")
                    
                    action = CustomerSupportAction(response=agent_response)
                    result = env.step(action)

                    step_obs = result.observation
                    reward = max(0.01, min(0.99, result.reward or 0.01))
                    rule_score = getattr(step_obs, "rule_score", 0.0)
                    done = result.done
                    error = None

                    grader_score = getattr(step_obs, "grader_score", 0.0)
                    next_stage = getattr(step_obs, "conversation_stage", current_stage)
                    stage_advanced = (next_stage != current_stage)

                    if done:
                        results["grader_score"] = grader_score

                    results["rule_scores"].append(rule_score)

                except Exception as exc:
                    print(f"[WARN] Step {step} failed: {exc}", file=sys.stderr, flush=True)
                    error = str(exc)
                    reward = 0.0
                    done = True
                    rule_score = 0.0
                    stage_advanced = False
                    next_stage = current_stage

                rewards.append(reward)
                log_step(
                    step=step,
                    action=agent_response,
                    reward=reward,
                    done=done,
                    error=error,
                )

                # 8. Update all robustness module states (internal, no API impact)
                if ResponseValidator and validator_state:
                    draft_intents = _get_draft_intents(agent_response, internal_task_name)
                    validator_state = ResponseValidator.update_state(
                        validator_state, agent_response, draft_intents, reward
                    )

                if AntiStallEngine and stall_state:
                    draft_intents = _get_draft_intents(agent_response, internal_task_name)
                    stall_state = AntiStallEngine.update(
                        stall_state, draft_intents, current_stage, stage_advanced
                    )

                # Update intent history
                history_intents.append(_get_draft_intents(agent_response, internal_task_name))
                
                # 9. EpisodeMemory: record high-reward responses
                if memory:
                    memory.record(internal_task_name, current_stage, agent_response, reward)

                # Update state for next iteration
                last_agent_response = agent_response
                last_reward = reward
                current_stage = next_stage

                if not done:
                    conversation_history = step_obs.conversation_history

                if done:
                    results["resolved"] = getattr(step_obs, "is_resolved", False) if "step_obs" in dir() else False
                    results["success"]  = getattr(step_obs, "success",     False) if "step_obs" in dir() else False
                    break

            # --- SUMMARY ---
            results["steps"]   = step
            results["rewards"] = rewards
            results["total_reward"] = sum(rewards)

            # Use grader_score (episode-level evaluation) as the primary score.
            # Fall back to avg step reward if grader_score is zero.
            final_grader = results.get("grader_score", 0.0)
            if final_grader > 0.0:
                raw_score = final_grader
            elif step > 0:
                raw_score = sum(rewards) / step
            else:
                raw_score = 0.01

            # Clamp to (0.01, 0.99) — validator rejects exactly 0.0 and 1.0
            results["score"] = max(0.01, min(0.99, raw_score))

            results["avg_rule_score"] = (
                sum(results["rule_scores"]) / len(results["rule_scores"])
                if results["rule_scores"] else 0.0
            )

            # Success: grader says so, or episode resolved, or score above threshold
            results["success"] = (
                results["score"] >= SUCCESS_SCORE_THRESHOLD
                or results["resolved"]
                or results.get("success", False)
            )

    except Exception as exc:
        print(f"[WARN] Client session failed: {exc}", file=sys.stderr, flush=True)

    log_end(
        success=results["success"],
        steps=results["steps"],
        score=results["score"],
        rewards=results["rewards"],
    )
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not API_KEY:
        print(
            "[WARN] No API key found (checked API_KEY and HF_TOKEN). "
            "LLM calls will use fallback responses.",
            file=sys.stderr,
            flush=True,
        )

    # Resolve where the environment server is / start it
    server_url = _resolve_server_url()

    if _TASK_FROM_EVALUATOR:
        # Hackathon evaluator injected a specific task — run only that one
        run_task(_TASK_FROM_EVALUATOR, server_url)
    else:
        # No task specified — run all 3 tasks (local verification)
        for task_id in TASKS_TO_RUN:
            run_task(task_id, server_url)


if __name__ == "__main__":
    main()
