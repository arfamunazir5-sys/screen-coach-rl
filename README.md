---
title: Screen Coach RL
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
tags:
  - openenv
---

# Digital Behavior Coach — OpenEnv Environment

An RL environment that simulates coaching a user toward healthier digital habits.
The agent observes screen usage patterns and takes actions to optimize wellbeing.

## Observation Space
- hour_of_day: int (0-23)
- screen_time_today: float (hours)
- app_category: str (social/work/entertainment)
- user_goal: str (focus/sleep/balance)
- streak_days: int
- last_action_effect: str
- resistance_level: float (0-1)
- habit_score: float (0-1)

## Action Space
- send_reminder, block_app, encourage, do_nothing

## Tasks
- easy: Reduce screen time for user exceeding 5 hours
- medium: Late night sleep intervention
- hard: Full day balance without over-intervening

## API Endpoints
- POST /reset — start new episode
- POST /step — take action
- GET /state — current state
- GET /ui — interactive demo

## Environment Variables
- API_BASE_URL (default: https://router.huggingface.co/v1)
- MODEL_NAME (default: meta-llama/Llama-3.1-8B-Instruct)
- HF_TOKEN (required)

## Baseline Scores
- easy: 0.72
- medium: 0.88
- hard: 0.86
