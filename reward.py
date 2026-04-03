import numpy as np

def calculate_reward(action, screen_time, hour, streak, user_goal):
    reward = 0

    # Too much screen time is bad
    if screen_time > 240:
        reward -= 2

    # Blocking app late at night is good
    if action == 2 and hour >= 22:
        reward += 3

    # Sending reminder before it gets too bad
    if action == 1 and screen_time > 180:
        reward += 2

    # Praising when user is doing well
    if action == 3 and screen_time < 120:
        reward += 2

    # Doing nothing when usage is very high is bad
    if action == 0 and screen_time > 300:
        reward -= 3

    # Reward streaks
    if streak > 3:
        reward += 1

    # Goal specific rewards
    # Goal 0 = focus, Goal 1 = sleep, Goal 2 = balance
    if user_goal == 1 and action == 2 and hour >= 21:
        reward += 2
    if user_goal == 0 and action == 1 and hour < 18:
        reward += 1

    # Gradual improvement reward
    if screen_time < 180:
        reward += 1

    return reward