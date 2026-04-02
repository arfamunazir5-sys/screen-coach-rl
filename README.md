# Screen Coach RL Environment

## What is this?
A reinforcement learning environment where an AI coach 
learns to reduce unhealthy screen habits.

## The Problem
People struggle with screen addiction and late-night scrolling.

## Our Solution
An AI agent that observes user behavior and decides the 
best intervention at each hour of the day.

## How it works
- State: time of day, screen usage, user goal, streak
- Actions: do nothing, remind, block, praise
- Reward: positive for reducing usage, negative for ignoring bad habits
## How to run
pip install gymnasium numpy
python main.py

## Team
- [Name 1] - Environment design
- [Name 2] - Reward function
- [Name 3] - Documentation & testing