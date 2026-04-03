from environment import DigitalCoachEnv, Action

def calculate_reward(screen_time: float, action_type: str, goal: str, hour: int) -> float:
    base = 0.0

    if action_type == "block_app" and screen_time > 6.0:
        base = 0.8
    elif action_type == "send_reminder" and screen_time > 4.0:
        base = 0.6
    elif action_type == "encourage":
        base = 0.4
    elif action_type == "do_nothing" and screen_time < 3.0:
        base = 0.5
    else:
        base = 0.1

    if goal == "sleep" and hour >= 22:
        base = min(1.0, base + 0.2)
    elif goal == "focus" and action_type == "block_app":
        base = min(1.0, base + 0.15)

    return round(base, 2)