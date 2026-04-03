# Digital Behavior Coach — OpenEnv Environment

## What It Does
Simulates a user's digital screen habits. An AI agent observes usage
patterns and takes actions to guide the user toward healthier behavior.
The user has realistic resistance to interventions and habit memory
that carries across episodes.

## Observation Space
| Field | Type | Description |
|---|---|---|
| hour_of_day | int | Current hour (0-23) |
| screen_time_today | float | Hours of screen use today |
| app_category | str | Type of app being used |
| user_goal | str | User's stated goal |
| streak_days | int | Consecutive good days |
| last_action_effect | str | Result of last action |
| resistance_level | float | How likely user is to ignore actions |
| habit_score | float | Overall habit health score |

## Action Space
- send_reminder — nudge the user
- block_app — forcibly restrict access
- encourage — reinforce positive streaks
- do_nothing — no intervention

## Tasks
- Easy: Remind a user with 5.5h screen time
- Medium: Late night intervention for sleep goal
- Hard: Full day management without over-intervening

## Setup
pip install -r requirements.txt
python main.py

## Run Grader
python grader.py

## Run Inference
export OPENAI_API_KEY=your_key
export MODEL_NAME=gpt-4o-mini
export API_BASE_URL=https://api.openai.com/v1
python inference.py

## Baseline Scores
| Task | Score |
|---|---|
| Easy | ~0.72 |
| Medium | ~0.68 |
| Hard | ~0.58 |