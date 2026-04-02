from environment import ScreenCoachEnv
from reward import calculate_reward

# Create the environment
env = ScreenCoachEnv()

# Run 5 test days
for episode in range(5):
    state, _ = env.reset()
    total_reward = 0
    done = False

    print(f"\n--- Day {episode + 1} ---")

    while not done:
        # AI picks a random action for now
        action = env.action_space.sample()

        # Take the action
        next_state, _, done, _, _ = env.step(action)

        # Calculate reward
        reward = calculate_reward(
            action,
            next_state[1],  # screen time
            next_state[0],  # hour
            next_state[3]   # streak
        )

        total_reward += reward
        state = next_state

    print(f"Total reward: {total_reward}")
    print(f"Final screen time: {state[1]:.0f} mins")