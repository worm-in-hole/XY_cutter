import gymnasium as gym
from gymnasium.utils.play import play

play(gym.make('LunarLander-v2', render_mode='rgb_array'),
    keys_to_action={'w': 2, 'a': 1, 'd': 3}, noop=0)
