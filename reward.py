def calculate_reward(action, screen_time, hour, streak):
    reward = 0

    # Bad: too much screen time
    if screen_time > 240:  # more than 4 hours
        reward -= 2

    # Good: blocking app during late night
    if action == 2 and hour >= 22:
        reward += 3

    # Good: sending reminder before it gets bad
    if action == 1 and screen_time > 180:
        reward += 2

    # Good: maintaining a streak
    if streak > 3:
        reward += 1

    # Bad: doing nothing when usage is very high
    if action == 0 and screen_time > 300:
        reward -= 3

    return reward