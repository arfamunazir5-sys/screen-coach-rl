---
title: Screen Coach RL
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# Digital Behavior Coach — OpenEnv Environment

## Overview
An RL environment that simulates a real user with screen addiction habits.
An AI agent learns to coach the user toward healthier digital behavior using
reminders, app blocking, and encouragement. The user has realistic resistance
to interventions and habit memory that carries across episodes.

## Why This Matters
Screen addiction is a real problem affecting millions. This environment allows
researchers to train and evaluate AI coaching agents in a safe simulation
before deploying to real users.

## Observation Space
| Field | Type | Description |
|---|---|---|
| hour_of_day | int | Current hour (0-23) |
| screen_time_today | float | Hours of screen use today |
| app_category | str | Type of app being used |
| user_goal | str | User's stated goal (focus/sleep/balance) |
| streak_days | int | Consecutive days of good behavior |
| last_action_effect | str | Result of last action taken |
| resistance_level | float | How likely user is to ignore interventions |
| habit_score | float | Overall habit health score (0-1) |

## Action Space
| Action | Description |
|---|---|
| send_reminder | Nudge the user to take a break |
| block_app | Forcibly restrict app access |
| encourage | Reinforce positive streaks |
| do_nothing | No intervention |

## Key Features
- User resistance — spamming reminders increases resistance and lowers reward
- Habit memory — past behavior affects future episodes
- Partial rewards — agent gets signal throughout episode not just at the end
- Late night bonuses — interventions near bedtime get extra reward

## Tasks
| Task | Difficulty | Description |
|---|---|---|
| Remind Over-User | Easy | Reduce screen time for user exceeding 5 hours |
| Sleep Intervention | Medium | Help user with sleep goal disengage late at night |
| Full Day Balance | Hard | Manage habits across full day without over-intervening |

## Setup
```bash
pip install -r requirements.txt
python main.py

## Baseline Scores
| Task | Difficulty | Score |
|---|---|---|
| easy | Easy | 0.33 |
| medium | Medium | 0.88 |
| hard | Hard | 0.86 |

## Run Inference
```bash
export HF_TOKEN=your_groq_key
python inference.py
```
