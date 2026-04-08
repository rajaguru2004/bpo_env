"""
MoodAdaptivePolicy — Emotional intelligence and tone accelerator (internal only).

Adjusts agent behavior based on detected customer mood.
Specifically accelerates resolution for 'angry' customers by forcing 
concise, action-first responses.
"""

from typing import Optional

class MoodAdaptivePolicy:
    """
    Adjusts the conversation strategy based on user mood.
    """

    @classmethod
    def get_mood_hint(cls, mood: str) -> Optional[str]:
        """
        Returns a style/strategy hint based on the detected mood.
        """
        if mood == "angry":
            return (
                "\n\n[MOOD_ADAPTIVE_STRICT]: The customer is ANGRY. "
                "1. Be extremely concise (2 sentences max). "
                "2. Provide IMMEDIATE RESOLUTION details now. "
                "3. Avoid long apologies, repetitive fluff, or generic empathy. "
                "4. Lead with action: 'I understand. I am fixing this now — here is what I am doing...'"
            )
        elif mood == "confused":
            return (
                "\n\n[MOOD_ADAPTIVE_GUIDE]: The customer is CONFUSED. "
                "Be patient and explanatory. Step-by-step guidance is preferred."
            )
        
        return None
