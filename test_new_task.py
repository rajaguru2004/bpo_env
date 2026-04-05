
import os
import sys

# Ensure root is in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

try:
    from server.bpo_env_environment import CustomerSupportEnvironment
    from models import CustomerSupportAction
except ImportError:
    from server.bpo_env_environment import CustomerSupportEnvironment
    from models import CustomerSupportAction

def test_subscription_management():
    print("Testing 'subscription_management' task...")
    env = CustomerSupportEnvironment()
    
    # 1. Reset with the new task
    obs, state = env.reset(task_name="subscription_management")
    
    print(f"Task Name: {obs.task_name}")
    print(f"Initial Message: {obs.customer_message}")
    print(f"Initial Stage: {obs.conversation_stage}")
    
    assert obs.task_name == "subscription_management"
    assert "subscription" in obs.customer_message.lower()
    assert obs.conversation_stage == "start"
    
    # 2. Mock a step (Verification stage)
    # The 'start' stage advances on 'greeting', 'apology', or 'de_escalation'
    action = CustomerSupportAction(response="I'm so sorry to hear that. I'd be happy to help you with your subscription.")
    obs, reward, done, state = env.step(action)
    
    print(f"Next Stage: {obs.conversation_stage}")
    print(f"Reward: {reward}")
    print(f"Customer Response: {obs.customer_message}")
    
    assert obs.conversation_stage == "verification"
    assert "SUBS-999" in obs.customer_message
    
    print("Verification successful!")

if __name__ == "__main__":
    try:
        test_subscription_management()
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)
