from environment import DigitalCoachEnv, Action

def run_demo():
    env = DigitalCoachEnv()
    obs = env.reset()

    print("=" * 50)
    print("DIGITAL BEHAVIOR COACH - DEMO")
    print("=" * 50)
    print(f"User Goal: {obs.user_goal}")
    print(f"Screen Time: {obs.screen_time_today} hours")
    print(f"Hour of Day: {obs.hour_of_day}")
    print(f"Resistance Level: {obs.resistance_level}")
    print(f"Habit Score: {obs.habit_score}")
    print("=" * 50)

    actions = [
        "send_reminder",
        "encourage",
        "block_app",
        "do_nothing",
        "send_reminder",
        "encourage"
    ]

    total_reward = 0.0
    for action_type in actions:
        action = Action(action_type=action_type)
        obs, reward, done, info = env.step(action)
        total_reward += reward.value
        print(f"Action: {action_type}")
        print(f"Reward: {reward.value} — {reward.reason}")
        print(f"Screen Time Now: {obs.screen_time_today} hrs | Resistance: {obs.resistance_level}")
        print("-" * 50)
        if done:
            break

    print(f"Total Reward: {round(total_reward, 2)}")
    print(f"Final Habit Score: {obs.habit_score}")
    print("=" * 50)

if __name__ == "__main__":
    run_demo()