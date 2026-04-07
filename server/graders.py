class BaseGrader:
    def grade(self, trajectory: dict) -> dict:
        raise NotImplementedError


class OrderStatusGrader(BaseGrader):
    def grade(self, trajectory):
        state = trajectory.get("state", {})
        intents = state.get("collected_intents", {})

        # The system uses tracking_info, delivery_info, empathy as keys in collected_intents
        required = ["empathy", "tracking_info", "delivery_info"]

        matched = sum(1 for i in required if intents.get(i, False))
        score = matched / len(required)

        return {
            "score": float(score),
            "reason": "Order status completion evaluation"
        }


class DamagedProductGrader(BaseGrader):
    def grade(self, trajectory):
        state = trajectory.get("state", {})
        intents = state.get("collected_intents", {})

        required = ["empathy", "information_request", "replacement"]

        matched = sum(1 for i in required if intents.get(i, False))
        score = matched / len(required)

        return {
            "score": float(score),
            "reason": "Damaged product resolution evaluation"
        }


class EscalationGrader(BaseGrader):
    def grade(self, trajectory):
        state = trajectory.get("state", {})
        intents = state.get("collected_intents", {})

        required = ["empathy", "escalation", "refund"]

        matched = sum(1 for i in required if intents.get(i, False))
        score = matched / len(required)

        return {
            "score": float(score),
            "reason": "Escalation handling evaluation"
        }
