import os
import json
from openai import OpenAI
from environment import DigitalCoachEnv, Action, Observation

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.1-8b-instant")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
client = OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)

VALID_ACTIONS = ["send_reminder", "block_app", "encourage", "do_nothing"]

def get_agent_action(observation: dict) -> str:
    prompt = f"""You are a digital behavior coach AI agent.

Current user state:
{json.dumps(observation, indent=2)}

Choose ONE action to help this user build healthier digital habits.
Valid actions: {VALID_ACTIONS}

Reply with ONLY the action name, nothing else."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20
    )
    action = response.choices[0].message.content.strip().lower()
    if action not in VALID_ACTIONS:
        action = "send_reminder"
    return action


def run_episode(task_name: str, scenario_override: dict = None):
    env = DigitalCoachEnv()
    obs = env.reset()

    if scenario_override:
        env.state_data.update(scenario_override)
        obs = Observation(**env.state_data)

    total_reward = 0.0
    step_num = 0

    print(json.dumps({"event": "START", "task": task_name, "observation": obs.model_dump()}))

    while True:
        action_type = get_agent_action(obs.model_dump())
        action = Action(action_type=action_type)
        obs, reward, done, info = env.step(action)
        total_reward += reward.value
        step_num += 1

        print(json.dumps({
            "event": "STEP",
            "step": step_num,
            "action": action_type,
            "reward": reward.value,
            "reason": reward.reason,
            "done": done
        }))

        if done:
            break

    final_score = round(total_reward / step_num, 4)
    print(json.dumps({
        "event": "END",
        "task": task_name,
        "total_reward": round(total_reward, 4),
        "steps": step_num,
        "score": final_score
    }))
    return final_score


if __name__ == "__main__":
    tasks = [
        ("easy",   {"screen_time_today": 5.5, "user_goal": "balance", "hour_of_day": 15}),
        ("medium", {"screen_time_today": 6.5, "user_goal": "sleep",   "hour_of_day": 22}),
        ("hard",   {"screen_time_today": 2.0, "user_goal": "focus",   "hour_of_day": 9}),
    ]

    all_scores = {}
    for name, scenario in tasks:
        score = run_episode(name, scenario)
        all_scores[name] = score

    print(json.dumps({"event": "FINAL_SCORES", "scores": all_scores}))
