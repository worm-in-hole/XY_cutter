import random
import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.buffer = []
        self._next_idx = 0

    def add(self, item):
        if len(self.buffer) > self._next_idx:
            self.buffer[self._next_idx] = item
        else:
            self.buffer.append(item)
        if self._next_idx == self.buffer_size - 1:
            self._next_idx = 0
        else:
            self._next_idx = self._next_idx + 1

    def sample(self, batch_size):
        indices = [random.randint(0, len(self.buffer) - 1) for _ in range(batch_size)]

        states = [self.buffer[i][0] for i in indices]
        actions = [self.buffer[i][1] for i in indices]
        rewards = [self.buffer[i][2] for i in indices]
        n_states = [self.buffer[i][3] for i in indices]
        dones = [self.buffer[i][4] for i in indices]
        return states, actions, rewards, n_states, dones

    def length(self):
        return len(self.buffer)


class OrnsteinUhlenbeckActionNoise:
    """
    Ornstein-Uhlenbeck noise implemented by OpenAI
    Copied from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
    """

    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = (self.x_prev +
             self.theta * (self.mu - self.x_prev) * self.dt +
             self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
             )
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
