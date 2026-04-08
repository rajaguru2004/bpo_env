"""
EpisodeMemory — Lightweight experience replay for biasing LLM generation
toward high-reward response patterns (internal only).

Stores (state_key, response_snippet, reward) tuples per task+stage.
Before each LLM call, retrieves the top-3 high-reward patterns for the
current task+stage and injects them as few-shot hints.

Implementation uses simple string-key matching (task_name + stage_name).
No embeddings required — zero external dependencies.

This module NEVER appears in API output or the Observation schema.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HIGH_REWARD_THRESHOLD = 0.65   # Only store responses with reward >= this
_TOP_K = 3                       # Number of examples to retrieve
_MAX_STORE_PER_KEY = 10          # Max examples stored per task+stage key
_SNIPPET_LENGTH = 200            # Max chars per stored snippet


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    state_key: str          # "{task_name}:{stage_name}"
    response_snippet: str   # Truncated response for few-shot
    reward: float
    stage_name: str
    task_name: str


# ---------------------------------------------------------------------------
# EpisodeMemory
# ---------------------------------------------------------------------------

class EpisodeMemory:
    """
    Lightweight memory store indexed by task+stage key.

    Thread-safety: NOT thread-safe (one instance per inference run).
    The memory builds up across episodes within a single inference session.
    """

    def __init__(self) -> None:
        # key → list of (reward, snippet) tuples, kept sorted descending
        self._store: Dict[str, List[Tuple[float, str]]] = {}
        self._total_stored: int = 0

    def _make_key(self, task_name: str, stage_name: str) -> str:
        return f"{task_name}:{stage_name}"

    def record(
        self,
        task_name: str,
        stage_name: str,
        response: str,
        reward: float,
    ) -> None:
        """
        Store a (response, reward) pair if reward >= threshold.
        Maintains top-N entries per key.
        """
        if reward < _HIGH_REWARD_THRESHOLD:
            return

        key = self._make_key(task_name, stage_name)
        snippet = response[:_SNIPPET_LENGTH].strip()

        if key not in self._store:
            self._store[key] = []

        entries = self._store[key]
        # Use a min-heap of (reward, snippet) to keep top-N
        if len(entries) < _MAX_STORE_PER_KEY:
            heapq.heappush(entries, (reward, snippet))
        elif reward > entries[0][0]:  # Better than worst stored
            heapq.heapreplace(entries, (reward, snippet))

        self._total_stored += 1

    def retrieve_examples(
        self,
        task_name: str,
        stage_name: str,
        top_k: int = _TOP_K,
    ) -> List[Tuple[float, str]]:
        """
        Return top-k (reward, snippet) pairs for the given task+stage.
        Returns empty list if no examples found.
        Sorted descending by reward.
        """
        key = self._make_key(task_name, stage_name)
        entries = self._store.get(key, [])
        if not entries:
            return []

        # Sort descending by reward (entries is a min-heap, so sort manually)
        sorted_entries = sorted(entries, key=lambda x: x[0], reverse=True)
        return sorted_entries[:top_k]

    def build_few_shot_hint(
        self,
        task_name: str,
        stage_name: str,
    ) -> Optional[str]:
        """
        Build a few-shot injection string for the LLM prompt.
        Prioritizes top-tier (reward > 0.9) examples.
        """
        examples = self.retrieve_examples(task_name, stage_name)
        if not examples:
            return None

        # Check if we have any ultra-high quality examples
        has_elite = any(r >= 0.9 for r, _ in examples)
        
        lines = [
            f"\n[PAST HIGH-SCORING RESPONSE PATTERNS for '{task_name}' / '{stage_name}' stage]:",
            "The following examples received high scores. Use them as style inspiration "
            "(do NOT copy verbatim — generate a fresh, diverse response):",
        ]
        
        if has_elite:
            lines[1] = (
                "The following EXCELLENT examples received top marks. Follow their "
                "structure and professional tone closely while using fresh phrasing:"
            )

        for i, (reward, snippet) in enumerate(examples, 1):
            star = " ★" if reward >= 0.9 else ""
            lines.append(f"  Example {i} (score={reward:.2f}){star}: \"{snippet}\"")

        lines.append(
            "Generate a response in a similar quality but with fresh phrasing."
        )
        return "\n".join(lines)

    def has_examples(self, task_name: str, stage_name: str) -> bool:
        """Quick check if there are any stored examples."""
        return bool(self._store.get(self._make_key(task_name, stage_name)))

    @property
    def total_stored(self) -> int:
        return self._total_stored

    def summary(self) -> Dict[str, int]:
        """Return count of stored examples per key (for debug logging)."""
        return {k: len(v) for k, v in self._store.items()}
