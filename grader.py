from environment import DigitalCoachEnv, Action

def run_task(task_name: str, scenario: dict, actions: list, success_fn) -> float:
    env = DigitalCoachEnv()
    env.reset()
    env.state_data.update(scenario)

    total_reward = 0.0
    steps = 0
    for action_type in actions:
        action = Action(action_type=action_type)
        _, reward, done, _ = env.step(action)
        total_reward += reward.value
        steps += 1
        if done:
            break

    score = success_fn(total_reward, steps, env.state_data)
    print(f"[TASK] {task_name} → Score: {score:.2f}")
    return score


def task_easy() -> float:
    scenario = {
        "hour_of_day": 15,
        "screen_time_today": 5.5,
        "app_category": "social",
        "user_goal": "balance",
        "streak_days": 2,
        "last_action_effect": "none",
        "resistance_level": 0.1,
        "habit_score": 0.5
    }
    actions = ["send_reminder", "send_reminder", "encourage"]

    def success(total_reward, steps, final_state):
        return min(1.0, total_reward / (steps * 0.6))

    return run_task("Easy: Reminder Task", scenario, actions, success)


def task_medium() -> float:
    scenario = {
        "hour_of_day": 22,
        "screen_time_today": 6.5,
        "app_category": "entertainment",
        "user_goal": "sleep",
        "streak_days": 1,
        "last_action_effect": "none",
        "resistance_level": 0.2,
        "habit_score": 0.4
    }
    actions = ["block_app", "send_reminder", "encourage", "do_nothing"]

    def success(total_reward, steps, final_state):
        return min(1.0, total_reward / (steps * 0.65))

    return run_task("Medium: Sleep Intervention", scenario, actions, success)


def task_hard() -> float:
    scenario = {
        "hour_of_day": 9,
        "screen_time_today": 2.0,
        "app_category": "work",
        "user_goal": "focus",
        "streak_days": 5,
        "last_action_effect": "none",
        "resistance_level": 0.3,
        "habit_score": 0.8
    }
    actions = [
        "do_nothing", "do_nothing", "encourage",
        "send_reminder", "block_app", "encourage",
        "do_nothing", "send_reminder", "block_app", "encourage"
    ]

    def success(total_reward, steps, final_state):
        streak_bonus = 0.1 if final_state["streak_days"] > 5 else 0.0
        return min(1.0, (total_reward / (steps * 0.55)) + streak_bonus)

    return run_task("Hard: Full Day Balance", scenario, actions, success)


if __name__ == "__main__":
    scores = {
        "easy": task_easy(),
        "medium": task_medium(),
        "hard": task_hard()
    }
    if __name__ == "__main__":
    scores = {
        "easy": task_easy(),
        "medium": task_medium(),
        "hard": task_hard()
    }
    print("=== GRADER RESULTS ===")
    for k, v in scores.items():
        print(f"  {k}: {v:.2f}")
