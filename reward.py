def calculate_reward(screen_time, action_type, goal, hour, streak, resistance, consecutive_reminders):
    base = 0.0
    is_work_hours = 9 <= hour <= 17
    is_late_night = hour >= 22
    is_evening = 18 <= hour <= 21

    if action_type == "send_reminder":
        if consecutive_reminders > 2:
            base = max(0.0, 0.1 - (consecutive_reminders * 0.02))
        elif screen_time > 6.0:
            base = 0.7
        elif screen_time > 4.0:
            base = 0.5
        else:
            base = 0.15

    elif action_type == "block_app":
        if is_work_hours:
            base = 0.05
        elif screen_time > 7.0:
            base = 0.9
        elif screen_time > 5.0:
            base = 0.7
        else:
            base = 0.1

    elif action_type == "encourage":
        if streak > 7:
            base = 0.9
        elif streak > 4:
            base = 0.75
        elif streak > 1:
            base = 0.5
        else:
            base = 0.2

    elif action_type == "do_nothing":
        if screen_time < 2.0:
            base = 0.6
        elif screen_time < 3.0:
            base = 0.45
        elif screen_time < 4.0:
            base = 0.2
        else:
            base = 0.0

    if is_late_night and goal == "sleep":
        base = min(1.0, base + 0.2)
    if is_evening and goal == "balance" and action_type == "send_reminder":
        base = min(1.0, base + 0.1)
    if screen_time > 8.0 and action_type in ["block_app", "send_reminder"]:
        base = min(1.0, base + 0.15)

    return round(base, 2)
