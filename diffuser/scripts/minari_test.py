import minari

import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

env = gym.make('PointMaze_UMaze-v3', render_mode="human")

# dataset = minari.load_dataset('D4RL/pointmaze/umaze-v2')
# env  = dataset.recover_environment(eval_env=True, render_mode="human")

observation = env.reset()

episode_over = False
while not episode_over:
    action = env.action_space.sample()  # policy currently randomly samples action space
    observation, reward, terminated, truncated, info = env.step(action)

    episode_over = terminated or truncated

env.close()