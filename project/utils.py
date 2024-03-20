import random
from typing import List

import numpy as np
import matplotlib.pyplot as plt


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

        states_digs = [self.buffer[i][0][:13] for i in indices]  # TODO: hardcode
        states_matr = [self.buffer[i][0][13:] for i in indices]
        actions = [self.buffer[i][1] for i in indices]
        rewards = [self.buffer[i][2] for i in indices]
        n_states_digs = [self.buffer[i][3][:13] for i in indices]
        n_states_matr = [self.buffer[i][3][13:] for i in indices]
        dones = [self.buffer[i][4] for i in indices]
        return states_digs, states_matr, actions, rewards, n_states_digs, n_states_matr, dones

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


def draw_training_plot(records):
    # Generate recent 50 interval average
    average_reward = []
    for idx in range(len(records)):
        avg_list = records[max(0, idx - 10):idx + 1]
        average_reward.append(np.average(avg_list))
    plt.plot(records, label='reward')
    plt.plot(average_reward, label='average reward')
    plt.xlabel('N steps')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()


def plot_density_chart(density_tbl: np.ndarray):
    plt.imshow(density_tbl, interpolation="nearest", origin="upper")
    plt.colorbar()
    plt.show()


def plot_head_path(polygon_coords: List[List[float]]):
    xs, ys = zip(*polygon_coords)  # create lists of x and y values
    plt.figure()
    plt.xlim(0, 30)
    plt.ylim(0, 30)
    plt.plot(xs, ys)
    plt.show()
