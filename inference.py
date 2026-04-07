import os
from openai import OpenAI
from environment import DigitalCoachEnv, Action, Observation
print("New file running")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
print("API KEY : ",API_KEY)
VALID_ACTIONS = ["send_reminder", "block_app", "encourage", "do_nothing"]

def get_agent_action(observation: dict) -> str:
    prompt = f"""You are a digital behavior coach AI agent.

Current user state:
{observation}

Choose ONE action to help this user build healthier digital habits.
Valid actions: {VALID_ACTIONS}

Reply with ONLY the action name, nothing else."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.7
        )
        action = (response.choices[0].message.content or "").strip().lower()
    except Exception:
        action = "send_reminder"

    if action not in VALID_ACTIONS:
        action = "send_reminder"

    return action


def run_episode(task_name: str, scenario_override: dict = None):
    env = DigitalCoachEnv()
    obs = env.reset()

    if scenario_override:
        env.state_data.update(scenario_override)
        obs = Observation(**env.state_data)

    rewards = []
    step_num = 0
    
    print(f"[START] task={task_name} env=screen-coach-rl model={MODEL_NAME}", flush=True)

    while True:
        action_type = get_agent_action(obs.model_dump())
        action = Action(action_type=action_type)

        result = env.step(action)
        obs, reward, done, info = result

        step_num += 1
        reward_val = reward.value if reward else 0.0
        rewards.append(reward_val)

        done_val = str(done).lower()

        print(
            f"[STEP] step={step_num} action={action_type} reward={reward_val:.2f} done={done_val} error=null",
            flush=True
        )

        if done:
            break

    total_reward = sum(rewards)
    score = total_reward / len(rewards) if rewards else 0.0
    score = min(max(score, 0.0), 1.0)

    success = score > 0.1
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} steps={step_num} score={score:.3f} rewards={rewards_str}",
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
