import copy
import math
from typing import Any, Union, Tuple, List
import numpy as np

import gymnasium as gym
import shapely
from shapely.geometry import Point, LineString
from shapely.geometry.polygon import Polygon


class XYCutter(gym.Env):
    """
    ## Description

    У нас в управлении есть 2D-станок с абстрактной обрабатывающей головкой.

    Все размеры задаются в миллиметрах, скорости - в м/с.

    На вход подаётся:
    - размер рабочей зоны
    - полигон с заготовкой (список с последовательными координатами углов)
    - список полигонов-отверстий внутри заготовки (их не надо обрабатывать)
    - пятно обрабатывающей головки (np.ndarray с интенсивностями обработки в каждом миллиметре) в единицу времени
    - желаемая средняя степень обработки
    - желаемая минимальная дисперсия между соседними зонами (допустимый перепад)


    ## Пространство действий

    Модель заточена под то, чтобы её выход было удобно переделывать в G-Code.

    Управление станком происходит путём генерации следующей точки, куда должна переместиться головка.

    Action = [next X, next Y, next Velocity, next isOn]

    next X - X-координата новой точки,
    next Y - Y-координата новой точки,
    next Velocity - скорость перемещения в новую точку (скорость влияет на интенсивность обработки),
    next isOn - 1, если головка совершает работу во время перемещения, 0 - головка выключена

    Note: Пока не учитываются импульсы, перегрузки и инерция.


    ## Пространство наблюдений

    State = [cur X, cur Y, cur Angle, cur Velocity, cur isOn, layer_painted, layer_loss]

    cur X - X-координата текущей точки
    cur Y - Y-координата текущей точки,
    cur Velocity - скорость, с которой головка пришла в текущую точку
    cur Angle - угол в радианах, откуда пришла головка в текущую точку
    cur isOn - флаг, что головка пришла в текущую точку в рабочем состоянии.

    layer_painted - 2D-массив с результатом сделанной работы
    layer_loss - 2D-массив с результатами напрасной работы (работа за пределами детали)

    Массивы считаются внутри функции step()


    ## Награда

    Площадь заготовки бьётся на квадратики 25*25 мм (свёртка 25*25 со страйдом 25*25?).
    Считается равномерность обработки внутри каждого квадратика (sigma, min, max).
    Считается равномерность обработки между соседними квадратиками (sigma).
    Считается min-max по детали, и по квадратикам.

    Идеальная ситуация, когда деталь обработана с заданной интенсивностью равномерно.
    Дисперсия между квадратиками не больше заданной.
    Дисперсия внутри квадратика не больше заданной.

    Есть состояние излишней обработки. Есть состояние недостаточной обработки.

    Штрафуем за включение обрабатывающей головки над деталью (сильно штрафуем).

    За обработку не над деталью тоже штрафуем, но не сильно.
    Можно просуммировать, и штрафовать в размере 5-10% от объёма работы.


    ## Начальное состояние

    Головка расположена в координатах (0, 0)
    Скорость = 0.
    Угол = 0.
    Матрицы полезной и потерянной работы - занулены.


    ## Конец эпизода

    Эпизод заканчивается, если:

    1. Работа выполнена (выполнены условия выше)
    2. Если точек больше 500 (?)
    3. Если площадь сделанной работы больше площади детали в 2 раз
       (можно в погонных метрах попробовать прикинуть, или во времени работы).


    ## Arguments

    ```python
    import gymnasium as gym
    gym.make('CartPole-v1')
    ```

    On reset, the `options` parameter allows the user to change the bounds used to determine
    the new random state.
    """

    def __init__(self,
                 working_area: LineString,
                 object_polys: List[Polygon],
                 protected_polys: List[Polygon],
                 processor_intensity: np.ndarray,
                 desired_intensity: Tuple[float, float],
                 ):
        """

        :param working_area: Рабочая зона станка, задана отрезком.
        :param object_polys: Список обрабатываемых деталей.
        :param protected_polys: Список зон, которые не нужно обрабатывать.
        :param processor_intensity: Матрица с интенсивностью обработки детали в 1 секунду.
          Если головка движется 1000 мм со скоростью 1 м/с (=1000мм/с),
          то данная матрица будет применена последовательно
          к 1000 точек (1000 мм/м) в зоне обработки с коэффициентом 0.001 (1 / 1000 м/с).
          Если головка 200 мм движется со скоростью 5 мм/с (=5000мм/с), то коэффициент составит 0.0002 (1 / 5000).
          Коэффициент для матрицы - это "экспозиция", сколько времени головка проводит над 1 мм (пикселем) детали.
          Обрабатывающая головка "смотрит" всегда в центр матрицы.
          Матрица может быть любой размерности.
        :param desired_intensity: Желаемый уровень средней работы и её дисперсии, выполненной над каждой точкой детали.
        """

        self._working_area: Polygon = Polygon.from_bounds(*working_area.bounds)
        """Рабочая зона станка, задана отрезком."""
        """Точка с минимальными координатами рабочей зоны (обычно - (0,0))."""
        self._object_polys: List[Polygon] = object_polys
        """Список обрабатываемых деталей."""
        self._protected_polys: List[Polygon] = protected_polys
        """Список зон, которые не нужно обрабатывать."""
        self._processor_intensity: np.ndarray = processor_intensity
        """Матрица с интенсивностью обработки детали в 1 секунду."""
        self._desired_intensity: Tuple[float, float] = desired_intensity
        """Желаемый уровень средней работы и её дисперсии, выполненной над каждой точкой детали."""

        self._cur_position = Point([0, 0])
        """Текущее положение головки (XY-случай, позиция в миллиметрах)."""
        self._cur_velocity = 0
        """Линейная скорость головки, по гипотенузе (м/сек)."""
        self._angle = 0
        """Угол движения головки (против часовой стрелки от оси OX, в градусах)."""
        self._is_on = 0
        """Обрабатывающая головка выключена."""

        self._dummy_nd_array: np.ndarray | None = None
        """Массив-шаблон (используется для копирования)."""
        self._detail_mask: np.ndarray | None = None
        """Маска из 0 и 1. 1, если точка принадлежит детали, 0 - если не принадлежит."""
        self._work_done: np.ndarray | None = None
        """Матрица размером с рабочую зону, которая аккумулирует работу головки над деталью, 
        пока головка движется во включенном состоянии."""
        self._work_in_vain: np.ndarray | None = None
        """Матрица размером с рабочую зону, которая аккумулирует работу головки за пределами детали, 
        пока головка движется во включенном состоянии."""

        # Проверки
        self._check_incoming_data()

        # Переходим в нулевое положение
        self.reset()

    def _check_incoming_data(self):
        """
        Проверяем входящие данные на корректность и здравый смысл.
        Падаем с AssertionError, если что-то не так.
        """
        assert self._working_area.bounds[0] >= 0, \
            'Рабочее поле не может начинаться в отрицательной зоне X.'
        assert self._working_area.bounds[1] >= 0, \
            'Рабочее поле не может начинаться в отрицательной зоне Y.'
        assert (self._working_area.bounds[2] - self._working_area.bounds[0]) > 0, \
            'Ширина рабочего поля не может быть нулевой или отрицательной.'
        assert (self._working_area.bounds[3] - self._working_area.bounds[1]) > 0, \
            'Длина рабочего поля не может быть нулевой или отрицательной.'

        assert all([len(poly.length) > 2 for poly in self._object_polys]), \
            'Обрабатываемый объект должен состоять минимум из 3 углов.'

        assert all([len(poly.length) > 2 for poly in self._protected_polys]), \
            'Каждая необрабатываемая область должна состоять минимум из 3 углов.'

        assert (self._processor_intensity.shape[0] > 0) and (self._processor_intensity.shape[0] % 2 == 1), \
            'Ширина обрабатываемой зоны должна быть положительной и нечётной.'
        assert (self._processor_intensity.shape[1] > 0) and (self._processor_intensity.shape[1] % 2 == 1), \
            'Длина обрабатываемой зоны должна быть положительной и нечётной.'
        assert self._processor_intensity.min() >= 0, \
            'Внутри обрабатываемой зоны не может делаться отрицательная работа.'

        assert self._desired_intensity[0] > 0, \
            'Желаемый средний уровень обработки не может быть нулевым или отрицательным.'
        assert self._desired_intensity[1] > 0, \
            'Дисперсия уровня обработки не может быть нулевой или отрицательной.'

    def reset(self,
              *,
              seed: int | None = None,
              options: dict[str, Any] | None = None):
        """
        Функция переводит модель в нулевое положение.
        Откуда начинается эпизод обучения или работы/инференса.

        :param seed: Не используется (у нас пока нет случайности).
        :param options: Не используется.
        :return: Возвращает вектор состояний.
        """

        self._cur_position = Point([0, 0])
        self._cur_velocity = 0
        self._angle = 0
        self._is_on = 0

        # Матрица float32, с которой мы будем делать копии с помощью `deepcopy.copy()`
        self._dummy_nd_array = np.ndarray(shape=(self._working_area.bounds[2] - self._working_area.bounds[0],
                                                 self._working_area.bounds[3] - self._working_area.bounds[1]),
                                          dtype=np.float32)
        self._dummy_nd_array.fill(0.0)

        # Маска из 0 и 1 в виде детали, с учётом возможных высечек.
        # С помощью неё будем определять, попала ли работа на деталь или мимо.
        self._detail_mask = np.ndarray(shape=(self._working_area.bounds[2] - self._working_area.bounds[0],
                                              self._working_area.bounds[3] - self._working_area.bounds[1]),
                                       dtype=np.uint8)

        # Заполняем маску: проверяем, что точка находится над деталью и не находится над выемкой.
        union_detail_poly = shapely.union_all(self._object_polys)
        union_protected_poly = shapely.union_all(self._protected_polys)
        detail_bounds = shapely.bounds(union_detail_poly)
        for i in range(int(self._working_area.bounds[0]), int(self._working_area.bounds[2])+1):
            for j in range(int(self._working_area.bounds[1]), int(self._working_area.bounds[3])+1):
                # Если мы над границами детали - начинаем проверять (для экономии CPU)
                if detail_bounds[0] <= i <= detail_bounds[2] and detail_bounds[1] <= j <= detail_bounds[3]:
                    point = Point(i, j)
                    chk1 = shapely.contains(union_detail_poly, point)
                    chk2 = True
                    if chk1 and self._protected_polys:
                        chk2 = not shapely.contains(union_protected_poly, point)
                    self._detail_mask[i, j] = 1 if chk1 and chk2 else 0

        # Полигон со степенью обработки каждой точки над областью детали
        self._work_done = copy.deepcopy(self._dummy_nd_array)

        # Полигон со степенью обработки каждой точки за пределами области детали (напрасная работа)
        self._work_in_vain = copy.deepcopy(self._dummy_nd_array)

        # Формируем вектор состояний (мы чуть позже изымем из него матрицы _work_done и _work_in_vain).
        state = [self._cur_position.coords[0],
                 self._cur_position.coords[1],
                 self._cur_velocity,
                 self._angle,
                 self._is_on,
                 self._work_done,
                 self._work_in_vain]
        return state

    def step(self, action: List):
        """
        Модель возвращает состояние среды после того, как агент сделал действие.

        :param action: Сделанное действие, представлено вектором значений
          [next X, next Y, next Velocity, next isOn].

        :return: Возвращает список из значений:
          [observation, reward, terminated, truncated, info]
        """
        next_x = max(self._working_area.bounds[0], min(action[0], self._working_area.bounds[2]))
        next_y = max(self._working_area.bounds[1], min(action[1], self._working_area.bounds[3]))
        next_velocity = action[2]  # скорость в м/с
        next_exposition = 1 / (next_velocity * 1000)  # время нахождения головки над 1 кв.мм. детали
        next_is_on = action[3]

        def _calc_angle(a, b):
            return math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1])
        # TODO: Что-то туплю... Надо посчитать угол между прошлым вектором и текущим.
        #   Это нужно для будущего расчёта перегрузок головки, если модель её на 180 градусов разворачивает.

        self._angle = 0

        # observation
        # Обновляем состояние матриц с полезной и напрасной работой.

        processor_matr_half_width = int(self._processor_intensity.shape[0] - 1 / 2)
        processor_matr_half_height = int(self._processor_intensity.shape[1] - 1 / 2)

        # Вычисляем работу, выполненную этим конкретным действием.
        unit_of_work_array = copy.deepcopy(self._dummy_nd_array)
        for i in range(self._cur_position.coords[0], next_x + 1):
            # Вычисляем, сколько колонок надо отрезать от матрицы self._processor_intensity
            # TODO: меня что-то рубит уже. Надо проверить.
            x_crop_left = min(0, i - processor_matr_half_width - self._working_area.bounds[0])
            x_crop_right = max(0, i + processor_matr_half_width - self._working_area.bounds[2])

            for j in range(self._cur_position.coords[1], next_y + 1):
                self._processor_intensity * next_exposition


        self.state = (x, x_dot, theta, theta_dot)
        reward = 0
        terminated = False

        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}


    def close(self):
        pass


