import sys
import os

# Add the current directory to sys.path to import from server
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from server.bpo_env_environment import (
    _compute_step_reward,
    classify_intents,
    check_completeness,
    check_sequence
)

def test_reward_scenarios():
    print("=== BPO Reward System v6 Test Scenarios ===\n")

    # Helper to simulate a reward calculation
    def get_reward(response, task="order_status", stage="inquiry", stage_idx=1, accepted=None):
        if accepted is None:
            accepted = {"information_provide", "information_request", "confirmation"}
        
        intents = classify_intents(response)
        intents_set = set(intents)
        
        reward, rule_score, comp_score, seq_score, reason = _compute_step_reward(
            response_text=response,
            intents=intents,
            intents_set=intents_set,
            accepted_intents=accepted,
            stage_advanced=False,
            is_repetitive=False,
            is_stalling=False,
            prev_step_was_wrong=False,
            task_name=task,
            stage_name=stage,
            stage_index=stage_idx
        )
        return {
            "reward": reward,
            "completeness": comp_score,
            "sequence": seq_score,
            "reason": reason,
            "intents": intents
        }

    # Scenario 1: Incomplete but correct intent
    print("Scenario 1: Incomplete response ('Your order is shipped')")
    res1 = get_reward("Your order is shipped.")
    print(f"Result: Reward={res1['reward']:.2f}, Completeness={res1['completeness']:.2f}, Sequence={res1['sequence']:.2f}")
    print(f"Reason: {res1['reason']}")
    assert res1['reward'] < 0.4, "Reward should be low for incomplete response"
    print("Status: PASS\n")

    # Scenario 2: Complete and correct response
    print("Scenario 2: Complete response (Tracking + Delivery Date)")
    full_resp = "Your order #12345 has been shipped. The tracking number is TRK987654321. You can expect delivery by April 3rd."
    res2 = get_reward(full_resp, stage="resolution", stage_idx=2)
    print(f"Result: Reward={res2['reward']:.2f}, Completeness={res2['completeness']:.2f}, Sequence={res2['sequence']:.2f}")
    print(f"Reason: {res2['reason']}")
    assert res2['reward'] > 0.7, "Reward should be high for complete response"
    print("Status: PASS\n")

    # Scenario 3: Sequence Violation
    print("Scenario 3: Sequence violation (Delivery date before tracking at inquiry stage)")
    seq_viol = "Your order will arrive on April 3rd. I don't have the tracking yet."
    res3 = get_reward(seq_viol, stage="inquiry", stage_idx=1)
    print(f"Result: Reward={res3['reward']:.2f}, Completeness={res3['completeness']:.2f}, Sequence={res3['sequence']:.2f}")
    print(f"Reason: {res3['reason']}")
    assert res3['sequence'] < 1.0, "Sequence score should be penalized"
    print("Status: PASS\n")

    # Scenario 4: Vague/Short Response
    print("Scenario 4: Vague/Short response ('We are checking')")
    res4 = get_reward("We are checking.")
    print(f"Result: Reward={res4['reward']:.2f}, Completeness={res4['completeness']:.2f}")
    print(f"Reason: {res1['reason']}")
    assert res4['completeness'] <= 0.3, "Short response should have capped completeness"
    print("Status: PASS\n")

    # Scenario 5: Damaged Product Empathy Check
    print("Scenario 5: Damaged Product (Empathy Stage - No apology)")
    res5 = get_reward("I see the item is broken. What is your order number?", 
                      task="damaged_product", stage="empathy", stage_idx=1,
                      accepted={"apology", "de_escalation", "information_request"})
    print(f"Result: Reward={res5['reward']:.2f}, Completeness={res5['completeness']:.2f}")
    print(f"Reason: {res5['reason']}")
    assert "Missing: apologize" in res5['reason'], "Should detect missing apology"
    print("Status: PASS\n")

    print("=== All Reward Tests Passed ===")

if __name__ == "__main__":
    try:
        test_reward_scenarios()
    except AssertionError as e:
        print(f"Test Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
