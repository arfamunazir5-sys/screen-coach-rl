import os
from openai import OpenAI
from environment import DigitalCoachEnv, Action, Observation

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

VALID_ACTIONS = ["send_reminder", "block_app", "encourage", "do_nothing"]
def get_agent_action(observation: dict) -> str:
    screen_time = observation["screen_time_today"]
    hour = observation["hour_of_day"]
    goal = observation["user_goal"]
    streak = observation["streak_days"]
    last = observation["last_action_effect"]

    prompt = f"""You are a digital behavior coach AI. Based on the user's current state, choose ONE action.

Current state:
- Screen time today: {screen_time} hours
- Hour of day: {hour} (0=midnight, 22=10pm)
- User goal: {goal}
- Streak days: {streak}
- Last action effect: {last}

Available actions: send_reminder, block_app, encourage, do_nothing

Rules:
- block_app for very high usage (>7h) or late night + sleep goal
- send_reminder for moderate usage (4-7h)
- encourage when streak is good (>4 days) and usage is low
- do_nothing when usage is healthy (<2h)

Reply with ONLY one of: send_reminder, block_app, encourage, do_nothing"""

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.0,
        )
        raw = completion.choices[0].message.content.strip().lower()
        for action in VALID_ACTIONS:
            if action in raw:
                return action
    except Exception as e:
        print(f"[DEBUG] LLM call failed: {e}", flush=True)

    # Fallback logic if LLM fails
    if hour >= 22 and goal == "sleep":
        return "block_app" if screen_time > 5 else "send_reminder"
    if screen_time > 7:
        return "block_app"
    if screen_time > 4:
        return "send_reminder" if last != "reminder_sent" else "encourage"
    if screen_time < 2:
        return "do_nothing"
    if streak > 4:
        return "encourage" if last != "encouraged" else "do_nothing"
    return "send_reminder"
def run_episode(task_name: str, scenario_override: dict = None):
    env = DigitalCoachEnv()
    obs = env.reset()

    if scenario_override:
        env.state_data.update(scenario_override)
        obs = Observation(**env.state_data)

    rewards = []
    step_num = 0
    score = 0.0
    success = False

    print(f"[START] task={task_name} env=screen-coach-rl model={MODEL_NAME}", flush=True)

    try:
        while True:
            action_type = get_agent_action(obs.model_dump())
            action = Action(action_type=action_type)

            obs, reward, done, info = env.step(action)

            step_num += 1
            reward_val = reward.value if reward else 0.0
            rewards.append(reward_val)

            print(
                f"[STEP] step={step_num} action={action_type} reward={reward_val:.2f} done={str(done).lower()} error=null",
                flush=True
            )

            if done:
                break

        MAX_REWARD_PER_STEP = 1.0
        total_reward = sum(rewards)
        max_possible = len(rewards) * MAX_REWARD_PER_STEP

        score = total_reward / max_possible if max_possible > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score > 0.1

    finally:
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(
            f"[END] success={str(success).lower()} steps={step_num} score={score:.2f} rewards={rewards_str}",
            flush=True
        )

    return score


if __name__ == "__main__":
    tasks = [
        ("easy",   {"screen_time_today": 5.5, "user_goal": "balance", "hour_of_day": 15}),
        ("medium", {"screen_time_today": 6.5, "user_goal": "sleep",   "hour_of_day": 22}),
        ("hard",   {"screen_time_today": 2.0, "user_goal": "focus",   "hour_of_day": 9}),
    ]

    for name, scenario in tasks:
        run_episode(name, scenario)