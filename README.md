-----

title: Screen Coach RL
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
tags:

- openenv

-----

# 🧠 Digital Behavior Coach — OpenEnv Environment

## Overview

An RL environment that simulates a real user with screen addiction habits.
An AI agent learns to coach the user toward healthier digital behavior using
reminders, app blocking, and encouragement. The user has realistic resistance
to interventions and habit memory that carries across episodes.

This environment is compliant with the [OpenEnv](https://github.com/openenv) specification
and tagged with `openenv` for automated discovery.

## Why This Matters

Screen addiction is a real problem affecting millions. This environment allows
researchers to train and evaluate AI coaching agents in a safe simulation
before deploying to real users.

## Observation Space

|Field             |Type |Description                                       |
|------------------|-----|--------------------------------------------------|
|hour_of_day       |int  |Current hour (0-23)                               |
|screen_time_today |float|Hours of screen use today                         |
|app_category      |str  |Type of app being used (social/work/entertainment)|
|user_goal         |str  |User’s stated goal (focus/sleep/balance)          |
|streak_days       |int  |Consecutive days of good behavior                 |
|last_action_effect|str  |Result of last action taken                       |
|resistance_level  |float|How likely user is to ignore interventions (0-1)  |
|habit_score       |float|Overall habit health score (0-1)                  |

## Action Space

|Action       |Description                   |
|-------------|------------------------------|
|send_reminder|Nudge the user to take a break|
|block_app    |Forcibly restrict app access  |
|encourage    |Reinforce positive streaks    |
|do_nothing   |No intervention               |

## Key Features

- **User resistance** — spamming reminders increases resistance and lowers reward
- **Habit memory** — past behavior affects future episodes
- **Partial rewards** — agent gets signal throughout episode, not just at the end
- **Late night bonuses** — interventions near bedtime get extra reward

## Tasks

|Task  |Difficulty|Description                                           |
|------|----------|------------------------------------------------------|
|easy  |Easy      |Reduce screen time for user exceeding 5 hours         |
|medium|Medium    |Help user with sleep goal disengage late at night     |
|hard  |Hard      |Manage habits across full day without over-intervening|

## API Endpoints

|Endpoint|Method|Description                                   |
|--------|------|----------------------------------------------|
|`/reset`|POST  |Reset environment, returns initial observation|
|`/step` |POST  |Take an action, returns obs/reward/done/info  |
|`/state`|GET   |Get current environment state                 |
|`/ui`   |GET   |Interactive visual demo                       |

## Setup & Usage

```bash
pip install -r requirements.txt
python app.py
```

## Run Grader

```bash
python grader.py
```

## Run Inference

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export HF_TOKEN=your_token_here
python inference.py
```

## Docker

```bash
docker build -t screen-coach-rl .
docker run -p 7860:7860 -e HF_TOKEN=your_token screen-coach-rl
```

## Environment Variables

|Variable    |Description              |Default                         |
|------------|-------------------------|--------------------------------|
|API_BASE_URL|LLM API endpoint         |https://router.huggingface.co/v1|
|MODEL_NAME  |Model identifier         |meta-llama/Llama-3.1-8B-Instruct|
|HF_TOKEN    |Your Hugging Face API key|(required, no default)          |

## Baseline Scores

|Task  |Score|
|------|-----|
|easy  |0.72 |
|medium|0.88 |
|hard  |0.86 |
