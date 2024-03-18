import torch
import torch.nn as nn
from torch.nn import functional as F


# Для расчёта конволюционных слоёв
# https://abdumhmd.github.io/files/conv2d.html

class PolicyNet(nn.Module):
    def __init__(self, params=5, matrixes=2500*3):
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

        # Где-то тут подмешиваем наши данные

        self.linear_mixer1 = nn.Linear(5+1014, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, 5)
        self.output = nn.Tanh()

    def forward(self, s_coords, s_matrixes):
        # А здесь на вход подаётся вектор из 4 элементов состояния
        outs = self.conv1(s_matrixes)
        outs = self.bn1(outs)
        outs = self.relu1(outs)
        outs = self.maxpool1(outs)
        outs = self.flatter(outs)  # выпрямили!

        # Подмешиваем управляющий вектор
        outs = torch.cat([s_coords, outs], dim=0)

        outs = self.linear_mixer1(outs)
        outs = self.linear2(outs)
        outs = self.linear3(outs)
        outs = self.output(outs)

        return outs



