"""
demo.py — Quick manual demo of the Digital Behavior Coach environment
Run: python demo.py
"""

from environment import DigitalCoachEnv, Action


def run_demo():
    env = DigitalCoachEnv()
    obs = env.reset()

    print("=" * 55)
    print("   DIGITAL BEHAVIOR COACH — DEMO")
    print("=" * 55)
    print(f"  User Goal:      {obs.user_goal}")
    print(f"  Screen Time:    {obs.screen_time_today} hours")
    print(f"  Hour of Day:    {obs.hour_of_day}:00")
    print(f"  Streak Days:    {obs.streak_days}")
    print(f"  Resistance:     {obs.resistance_level}")
    print(f"  Habit Score:    {obs.habit_score}")
    print("=" * 55)

    actions = [
        "send_reminder",
        "encourage",
        "block_app",
        "do_nothing",
        "send_reminder",
        "encourage",
    ]

    total_reward = 0.0
    for action_type in actions:
        action = Action(action_type=action_type)
        obs, reward, done, info = env.step(action)
        total_reward += reward.value
        print(f"Step {info['step']:>2} | {action_type:<16} | reward={reward.value:.2f} | {reward.reason}")
        if done:
            break

    print("=" * 55)
    print(f"  Total Reward:   {round(total_reward, 2)}")
    print(f"  Final Habit:    {obs.habit_score}")
    print("=" * 55)


if __name__ == "__main__":
    run_demo()
