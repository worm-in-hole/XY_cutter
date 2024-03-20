import time
from typing import List

import torch
from shapely import LineString, Polygon
from tqdm import tqdm

from environment import XYCutter
import numpy as np
from torch.nn import functional as F, Parameter

from project.model import QNet, PolicyNet
from project.utils import OrnsteinUhlenbeckActionNoise, ReplayBuffer, draw_training_plot, plot_density_chart, \
    plot_head_path

# Ссылки:
# https://spinningup.openai.com/en/latest/spinningup/rl_intro.html
# https://spinningup.openai.com/en/latest/algorithms/ddpg.html
# https://github.com/sshish/RL-DDPG/blob/master/main.ipynb
# ---
# https://github.com/Lornatang/ResNet-PyTorch/blob/main/model.py
# https://neurohive.io/ru/vidy-nejrosetej/resnet-34-50-101/
# https://education.yandex.ru/handbook/ml
# https://shapely.readthedocs.io/en/stable/reference/shapely.Polygon.html
# https://python-graph-gallery.com/2d-density-plot/

# Задали интенсивность обработки головкой
header_intensity = np.array(
    [
        [0.00, 0.00, 0.03, 0.05, 0.03, 0.00, 0.00],
        [0.00, 0.03, 0.10, 0.13, 0.10, 0.03, 0.00],
        [0.01, 0.03, 0.13, 0.18, 0.13, 0.03, 0.01],
        [0.01, 0.05, 0.18, 0.20, 0.18, 0.05, 0.01],
        [0.01, 0.03, 0.13, 0.18, 0.13, 0.03, 0.01],
        [0.00, 0.03, 0.10, 0.13, 0.10, 0.03, 0.00],
        [0.00, 0.00, 0.03, 0.05, 0.03, 0.00, 0.00],
    ], dtype=np.float32)

# Инициализировали окружение
env = XYCutter(
    working_area=LineString([[0, 0], [30, 30]]),
    object_polys=[Polygon([[10, 10], [10, 20], [20, 20], [20, 10], [10, 10]])],
    protected_polys=[Polygon([[12, 12], [12, 15], [15, 15], [15, 12], [12, 12]])],
    processor_intensity=header_intensity,
    desired_intensity=(0.008, 0.009),
)

plot_density_chart(header_intensity)

# выбрали устройство вычисления
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

max_values = torch.tensor([30, 30, 3, 1]).to(device)  # TODO: hardcode

# Сделали много Mu-сетей (Акторы, прогнозируют действие при заданном состоянии).
mu_origin_model = PolicyNet(max_values=max_values).to(device)  # mu_theta
mu_target_model = PolicyNet(max_values=max_values).to(device)  # mu_theta'
_ = mu_target_model.requires_grad_(False)  # target model doesn't need grad

# Сделали много Q-сетей (Twin-Q) (Критики, прогнозируют награду при заданном состоянии и действии).
q_origin_model1 = QNet().to(device)  # Q_phi1
q_origin_model2 = QNet().to(device)  # Q_phi2
q_target_model1 = QNet().to(device)  # Q_phi1'
q_target_model2 = QNet().to(device)  # Q_phi2'
_ = q_target_model1.requires_grad_(False)  # target model doesn't need grad
_ = q_target_model2.requires_grad_(False)  # target model doesn't need grad

GAMMA = 0.99
opt_q1 = torch.optim.AdamW(q_origin_model1.parameters(), lr=0.0005)
opt_q2 = torch.optim.AdamW(q_origin_model2.parameters(), lr=0.0005)
opt_mu = torch.optim.AdamW(mu_origin_model.parameters(), lr=0.0005)


def optimize(states_digs: List, states_matr: List, actions: List, rewards: List, next_states_digs: List, next_states_matr: List,dones: List):
    """
    Функция обучает "грубые" модели одного Актора и двух Критиков.

    :param states: список state'ов. Каждый state - список из 13 float'ов и 3 матриц.
    :param actions: список action'ов. Каждый action - список из 4 float'ов.
    :param rewards: список float'ов. Каждый float - отдельный reward.
    :param next_states: аналогично states.
    :param dones: список из bool'ов. Каждый bool - это флаг завершения покраски.
    :return:
    """
    # Преобразуем значения из окружающей среды в тензоры
    states_digs = torch.tensor(states_digs, dtype=torch.float).to(device)
    states_matr = torch.tensor(np.stack(states_matr, axis=0), dtype=torch.float).to(device)
    actions = torch.tensor(actions, dtype=torch.float).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float).to(device)
    next_states_digs = torch.tensor(next_states_digs, dtype=torch.float).to(device)
    next_states_matr = torch.tensor(np.stack(next_states_matr, axis=0), dtype=torch.float).to(device)
    dones = torch.tensor(dones, dtype=torch.float).to(device)

    # actions = actions.unsqueeze(dim=1)
    rewards = rewards.unsqueeze(dim=1)
    dones = dones.unsqueeze(dim=1)

    # Compute reward + gamma * (1 - done) * min Q (mu_target1(next_states), mu_target2(next_states))
    mu_tgt_next_actions = mu_target_model(next_states_digs, next_states_matr)  # "точная" модель
    # считаем min Q из двух сеток
    q1_tgt_next = q_target_model1(next_states_digs, next_states_matr, mu_tgt_next_actions)  # "точная" модель
    q2_tgt_next = q_target_model2(next_states_digs, next_states_matr, mu_tgt_next_actions)  # "точная" модель
    q_tgt_next_min = torch.minimum(q1_tgt_next, q2_tgt_next)
    q_tgt = rewards + GAMMA * (1.0 - dones) * q_tgt_next_min  # некая усреднённая награда для обеих сетей-критиков

    # Обучаем критика #1 (Q-сеть)
    opt_q1.zero_grad()
    q1_org = q_origin_model1(states_digs, states_matr, actions)  # "грубая" модель
    loss_q1 = F.mse_loss(
        q1_org,
        q_tgt,
        reduction="none")
    loss_q1.sum().backward()
    opt_q1.step()

    # Обучаем критика #2 (Q-сеть)
    opt_q2.zero_grad()
    q2_org = q_origin_model2(states_digs, states_matr, actions)  # "грубая" модель
    loss_q2 = F.mse_loss(
        q2_org,
        q_tgt,
        reduction="none")
    loss_q2.sum().backward()
    opt_q2.step()

    # Обучаем актора (Policy-сеть)
    opt_mu.zero_grad()
    mu_org = mu_origin_model(states_digs, states_matr)  # "грубая" модель Актора
    for p in q_origin_model1.parameters():
        p.requires_grad = False  # disable grad in q_origin_model1 before computation
    q_tgt_max = q_origin_model1(states_digs, states_matr, mu_org)  # это уже обученная "грубая" модель критика
    # TODO: не понимаю, что здесь происходит.
    (-q_tgt_max).sum().backward()
    opt_mu.step()
    for p in q_origin_model1.parameters():
        p.requires_grad = True  # enable grad again


TAU = 0.002
"""Коэффициент, с которым мы добавляем веса "грубой" модели к весам target-модели."""


def update_target():
    """
    Функция для мягкого обновления весов в target-моделях
    """
    var: Parameter
    var_target: Parameter
    for var, var_target in zip(q_origin_model1.parameters(), q_target_model1.parameters()):
        var_target.data = TAU * var.data + (1.0 - TAU) * var_target.data
    for var, var_target in zip(q_origin_model2.parameters(), q_target_model2.parameters()):
        var_target.data = TAU * var.data + (1.0 - TAU) * var_target.data
    for var, var_target in zip(mu_origin_model.parameters(), mu_target_model.parameters()):
        var_target.data = TAU * var.data + (1.0 - TAU) * var_target.data


# Выбираем действие с учётом шума Ornstein-Uhlenbeck
def calc_action_with_noise(state: List):
    with torch.no_grad():
        state_digs = np.array(state[:13])
        state_matr = np.array(state[13:])

        # добавляем новую ось - ось батчей
        s_digs_batch = np.expand_dims(state_digs, axis=0)
        s_digs_batch = torch.tensor(s_digs_batch, dtype=torch.float).to(device)
        s_matr_batch = np.expand_dims(state_matr, axis=0)
        s_matr_batch = torch.tensor(s_matr_batch, dtype=torch.float).to(device)

        # действие по модели (deterministic)
        action_det = mu_origin_model(s_digs_batch, s_matr_batch)  # "грубая" модель Актора
        action_det = action_det.squeeze(dim=1)  # избавились от оси батчей?
        # вычисляем шум (каждый раз новый)
        noise = ou_action_noise()
        # действие + небольшая случайность
        action = action_det.cpu().numpy() + noise  # шум небольшой, масштабировать не будем
        action = np.clip(action, [0.01, 0.01, 0.05, 0], [29.99, 29.99, 3, 1])  # TODO: hardcode
        return action[0].tolist()


# сбрасываем веса
ou_action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(1), sigma=np.ones(1) * 0.05)

# чистим буфер
buffer = ReplayBuffer(buffer_size=50000)

# start training
batch_size = 250  # 250
reward_records = []
cum_reward = 0
avg_reward = 0
progress_bar = tqdm(range(30))
for i in progress_bar:
    progress_bar.set_description('cum={0:0>5.1f}, avg={1:0>5.1f}'.format(cum_reward, avg_reward))
    # Run episode till done
    state = env.reset()
    done = False
    terminated = False
    cum_reward = 0
    cnt = 0
    while not done and not terminated:
        # для текущего состояния вычисляем оптимальное действие
        action = calc_action_with_noise(state)
        # забираем из окружения новое состояние среды
        state_next, reward, terminated, done, _ = env.step(action)  # TODO: нас интересует награда за шаг
        # кладём в буфер
        buffer.add([state, action, reward, state_next, float(done)])
        cum_reward += reward

        # Как только заполнили буфер до размера батча
        if buffer.length() >= batch_size:
            # забираем из буфера случайную выборку нужного размера
            states_digs, states_matr, actions, rewards, n_states_digs, n_states_matr, dones = buffer.sample(batch_size)
            # по выборке учим модельки
            optimize(states_digs, states_matr, actions, rewards, n_states_digs, n_states_matr, dones)
            # мягко обновляем веса в конечных модельках
            update_target()
        state = state_next
        cnt += 1
        # if cnt % 100 == 0:
        #     print('cum={0:0>5.1f}, avg={1:0>5.1f}'.format(cum_reward, avg_reward))
        if terminated or done:
            # plot_density_chart(state[13])
            # plot_density_chart(state[14])
            # plot_density_chart(state[15])
            plot_head_path([(x, y) for x,y,_,_ in env._actions])

            # Записываем награду эпизода в вектор с историей
    reward_records.append(cum_reward)
    # Считаем среднюю награду за последние 10 эпизодов
    avg_reward = np.average(reward_records[-10:])

    # Завершаем обучение, когда средняя награда держится выше 475.0
    if avg_reward > 200.0:
        break

draw_training_plot(reward_records)
time.sleep(0.1)
plot_density_chart(env._work_done)
time.sleep(0.1)
plot_density_chart(env._work_in_vain)

print("\nDone")
