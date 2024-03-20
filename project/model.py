import torch
import torch.nn as nn


# Для расчёта конволюционных слоёв
# https://abdumhmd.github.io/files/conv2d.html

class PolicyNet(nn.Module):
    """
    Политика, это функция принимающая решение.
    Набор правил. Стратегия. "Мозг" агента.

    В нашем случае, политика по текущему состоянию окружающей среды должна предсказывать оптимальное действие.
    """
    def __init__(self, state_digits_size=13, state_matrixes=50 * 50 * 3, act_dim=4):
        super().__init__()

        conv_channels = 6

        # Кусок про обработку детали (взято из ResNet).
        # Обрабатываем матрицы состояния.
        # 50*50*3 - размеры рабочей области на момент обучения. Но хочется уйти на произвольные размеры.
        self.conv1 = nn.Conv2d(3, conv_channels, (7, 7), (2, 2), (3, 3), bias=False)
        # 25*25*6
        self.bn1 = nn.BatchNorm2d(conv_channels)
        self.relu1 = nn.ReLU(True)
        self.maxpool1 = nn.MaxPool2d((3, 3), (2, 2), (1, 1))
        # 13*13*6 = 1014

        # Выпрямляем
        self.flatter = nn.Flatten()

        # Тут подмешиваем вектор с числами состояния (их у нас пока 5: текущее положение, скорость и т.д.)
        self.linear_mixer1 = nn.Linear(1014 + state_digits_size, 512)
        self.linear2 = nn.Linear(512, 128)
        self.output = nn.Linear(128, act_dim)
        # self.output = nn.Tanh()  # вектор действий пока состоит из 5 координат

    def forward(self, state_digits, state_matrixes):
        # А здесь на вход подаётся вектор из 4 элементов состояния
        outs = self.conv1(state_matrixes)
        outs = self.bn1(outs)
        outs = self.relu1(outs)
        outs = self.maxpool1(outs)
        outs = self.flatter(outs)  # выпрямили!

        # Подмешиваем управляющий вектор
        outs = torch.cat([outs, state_digits], dim=1)

        outs = self.linear_mixer1(outs)
        outs = self.linear2(outs)
        outs = self.output(outs)

        return outs


class QNet(nn.Module):
    """
    Функция ценности состояния-действия.
    Это средняя/суммарная награда агента в состоянии `s`, если он сделает действие `a` (не обязательно оптимальное),
    а потом будет действовать согласно оптимальной политике.
    """
    def __init__(self, state_digits=13, state_matrixes=50*50*3, act_dim=4):
        super().__init__()

        conv_channels = 6

        # Кусок про обработку детали
        # 50*50*3
        self.conv1 = nn.Conv2d(3, conv_channels, (7, 7), (2, 2), (3, 3), bias=False)
        # 25*25*6
        self.bn1 = nn.BatchNorm2d(conv_channels)
        self.relu1 = nn.ReLU(True)
        self.maxpool1 = nn.MaxPool2d((3, 3), (2, 2), (1, 1))
        # 13*13*6 = 1014

        # Выпрямляем
        self.flatter = nn.Flatten()

        # Тут подмешиваем вектор с числами состояния (их у нас пока 5: текущее положение, скорость и т.д.)
        self.linear_mixer1 = nn.Linear(1014 + state_digits + act_dim, 512)
        self.linear2 = nn.Linear(512, 128)
        self.output = nn.Linear(128, 1)  # Прогнозируется награда - одно число

    def forward(self, state_digits, state_matrixes, action):
        # А здесь на вход подаётся вектор из 4 элементов состояния
        outs = self.conv1(state_matrixes)
        outs = self.bn1(outs)
        outs = self.relu1(outs)
        outs = self.maxpool1(outs)
        outs = self.flatter(outs)  # выпрямили!

        # Подмешиваем управляющий вектор + вектор действий
        outs = torch.cat([outs, state_digits, action], dim=1)

        outs = self.linear_mixer1(outs)
        outs = self.linear2(outs)
        outs = self.output(outs)  # должна быть только одна циферка на выходе - прогнозируемая награда

        return outs
