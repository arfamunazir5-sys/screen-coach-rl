from pydantic import BaseModel
from typing import Optional
import random

class Observation(BaseModel):
    hour_of_day: int
    screen_time_today: float
    app_category: str
    user_goal: str
    streak_days: int
    last_action_effect: str
    resistance_level: float
    habit_score: float

class Action(BaseModel):
    action_type: str
    message: Optional[str] = ""

class Reward(BaseModel):
    value: float
    reason: str

class DigitalCoachEnv:
    def __init__(self):
        self.state_data = {}
        self.step_count = 0
        self.max_steps = 20
        self.consecutive_reminders = 0
        self.habit_history = []

    def reset(self) -> Observation:
        self.step_count = 0
        self.consecutive_reminders = 0
        past_avg = sum(self.habit_history[-7:]) / len(self.habit_history[-7:]) if self.habit_history else 4.0
        self.state_data = {
            "hour_of_day": random.randint(8, 22),
            "screen_time_today": round(min(12.0, past_avg + random.uniform(-1.0, 1.0)), 1),
            "app_category": random.choice(["social", "entertainment", "work"]),
            "user_goal": random.choice(["focus", "sleep", "balance"]),
            "streak_days": random.randint(0, 7),
            "last_action_effect": "none",
            "resistance_level": round(random.uniform(0.0, 0.3), 2),
            "habit_score": round(max(0.0, 1.0 - (past_avg / 12.0)), 2)
        }
        return Observation(**self.state_data)

    def step(self, action: Action):
        self.step_count += 1
        obs, reward_val, reason = self._apply_action(action)
        self.habit_history.append(self.state_data["screen_time_today"])
        done = self.step_count >= self.max_steps
        reward = Reward(value=reward_val, reason=reason)
        info = {"step": self.step_count}
        return obs, reward, done, info

    def state(self) -> dict:
        return self.state_data

    def _apply_action(self, action: Action):
        hour = self.state_data["hour_of_day"]
        screen_time = self.state_data["screen_time_today"]
        goal = self.state_data["user_goal"]
        resistance = self.state_data["resistance_level"]
        reward = 0.0
        reason = ""

        ignored = random.random() < resistance

        if action.action_type == "send_reminder":
            self.consecutive_reminders += 1
            if self.consecutive_reminders > 2:
                self.state_data["resistance_level"] = min(1.0, resistance + 0.1)
                reward = 0.1
                reason = "User ignoring repeated reminders, resistance increased"
            elif ignored:
                reward = 0.15
                reason = "Reminder ignored due to user resistance"
            elif screen_time > 4.0:
                self.state_data["screen_time_today"] = max(0, screen_time - 0.5)
                reward = 0.6
                reason = "Reminder helped reduce screen time"
            else:
                reward = 0.2
                reason = "Reminder sent but screen time was already low"
            self.state_data["last_action_effect"] = "reminder_sent"

        elif action.action_type == "block_app":
            self.consecutive_reminders = 0
            if ignored:
                reward = 0.2
                reason = "Block attempt resisted by user"
            elif screen_time > 6.0:
                self.state_data["screen_time_today"] = max(0, screen_time - 1.0)
                self.state_data["resistance_level"] = max(0.0, resistance - 0.05)
                reward = 0.8
                reason = "App blocked during heavy usage"
            else:
                reward = 0.1
                reason = "App blocked unnecessarily"
            self.state_data["last_action_effect"] = "app_blocked"

        elif action.action_type == "encourage":
            self.consecutive_reminders = 0
            if self.state_data["streak_days"] > 3:
                self.state_data["streak_days"] += 1
                self.state_data["resistance_level"] = max(0.0, resistance - 0.05)
                reward = 0.7
                reason = "Encouragement boosted existing streak"
            else:
                reward = 0.3
                reason = "Encouragement given but streak was short"
            self.state_data["last_action_effect"] = "encouraged"

        elif action.action_type == "do_nothing":
            self.consecutive_reminders = 0
            if screen_time < 3.0:
                reward = 0.5
                reason = "Correctly did nothing, usage was healthy"
            else:
                reward = 0.0
                reason = "Did nothing during high usage"
            self.state_data["last_action_effect"] = "no_action"

        if goal == "sleep" and hour >= 22 and action.action_type != "do_nothing":
            reward = min(1.0, reward + 0.2)
            reason += " + late night sleep bonus"

        self.state_data["hour_of_day"] = min(23, hour + 1)
        self.state_data["habit_score"] = round(max(0.0, 1.0 - (self.state_data["screen_time_today"] / 12.0)), 2)
        return Observation(**self.state_data), round(reward, 2), reason

env = DigitalCoachEnv()