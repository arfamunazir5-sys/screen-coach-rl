import gymnasium as gym
import numpy as np

class ScreenCoachEnv(gym.Env):
    def __init__(self):
        # What the AI can SEE (the situation)
        # [hour of day, screen time used, user goal, streak count]
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([23, 480, 3, 30]),
            dtype=np.float32
        )

        # What the AI can DO (4 choices)
        # 0 = do nothing, 1 = send reminder
        # 2 = block app, 3 = send praise
        self.action_space = gym.spaces.Discrete(4)

    def reset(self):
        # Start a new day for the user
        self.hour = 8          # 8am
        self.screen_time = 0   # no usage yet
        self.user_goal = 1     # 1 = focus goal
        self.streak = 0        # no streak yet
        return self._get_state(), {}

    def _get_state(self):
        return np.array([
            self.hour,
            self.screen_time,
            self.user_goal,
            self.streak
        ], dtype=np.float32)

    def step(self, action):
        # Simulate one hour passing
        self.hour += 1

        # User randomly uses screen (we simulate this)
        usage = np.random.randint(10, 60)

        # If AI blocked the app, usage is reduced
        if action == 2:
            usage = usage * 0.3

        self.screen_time += usage
        done = self.hour >= 23  # day ends at 11pm
        return self._get_state(), 0, done, False, {}