import math
from typing import Any, Union, Tuple, List
import numpy as np

import gymnasium as gym
import shapely
from shapely.geometry import Point
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

    Action = [next X, next Y, next Speed, next isOn]

    next X - X-координата новой точки,
    next Y - Y-координата новой точки,
    next Speed - скорость перемещения в новую точку (скорость влияет на интенсивность обработки),
    next isOn - 1, если головка совершает работу во время перемещения, 0 - головка выключена

    Note: Пока не учитываются импульсы, перегрузки и инерция.


    ## Пространство наблюдений

    State = [cur X, cur Y, cur Angle, cur Speed, cur isOn, layer_painted, layer_loss]

    cur X - X-координата текущей точки
    cur Y - Y-координата текущей точки,
    cur Speed - скорость, с которой головка пришла в текущую точку
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
                 max_coords_point: Point,
                 object_polys: List[Polygon],
                 protected_polys: List[Polygon],
                 processor_intensity: np.ndarray,
                 desired_intensity: Tuple[float, float],
                 ):
        """

        :param max_coords_point: Точка с максимальными координатами рабочей зоны.
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

        self._max_coords_point: Point = max_coords_point
        """Точка с максимальными координатами рабочей зоны."""
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
        assert self._max_coords_point.coords[0] > 0, 'Ширина рабочего поля не может быть нулевой или отрицательной.'
        assert self._max_coords_point.coords[1] > 0, 'Длина рабочего поля не может быть нулевой или отрицательной.'

        assert all([len(poly.length) > 2 for poly in self._object_polys]), \
            'Обрабатываемый объект должен состоять минимум из 3 углов.'

        assert all([len(poly.length) > 2 for poly in self._protected_polys]), \
            'Каждая необрабатываемая область должна состоять минимум из 3 углов.'

        assert self._processor_intensity.shape[0] > 0, \
            'Ширина обрабатываемой зоны должна быть положительной.'
        assert self._processor_intensity.shape[1] > 0, \
            'Длина обрабатываемой зоны должна быть положительной.'
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

        # Массив, с которого мы будем делать копии с помощью `deepcopy.copy()`
        self._dummy_nd_array = np.ndarray(shape=(self._field_width, self._field_height))
        self._dummy_nd_array.fill(0)

        # Маска из 0 и 1 в виде детали, с учётом возможных высечек.
        # С помощью неё будем определять, попала ли работа на деталь или мимо.
        self._detail_mask = np.ndarray(shape=(self._field_width, self._field_height))
        self._detail_mask.fill(0)

        detail_bounds = self._object_polys.bounds
        for i in range(self._field_width):
            for j in range(self._field_height):
                # Если мы над границами детали - начинаем проверять (для экономии CPU)
                if detail_bounds[0] <= i <= detail_bounds[2] and detail_bounds[1] <= j <= detail_bounds[3]:
                    point = Point(i, j)
                    chk1 = shapely.contains(self._object_polys, point)
                    chk2 = True
                    if chk1 and self._protected_polys:
                        chk2 = all([not shapely.contains(poly, point) for poly in self._protected_polys])
                    self._detail_mask[i, j] = 1 if chk1 and chk2 else 0

        # Полигон со степенью обработки каждой точки над областью детали
        self._work_done = np.ndarray(shape=(self._field_width, self._field_height))
        self._work_done.fill(0)

        # Полигон со степенью обработки каждой точки за пределами области детали (напрасная работа)
        self._work_in_vain = np.ndarray(shape=(self._field_width, self._field_height))
        self._work_in_vain.fill(0)

        state = [self._cur_x, self._cur_y, self._cur_velocity, self._angle, self._is_on,
                 self._work_done, self._work_in_vain]
        return state

    def step(self, action: List):
        # Action = [next X, next Y, next Speed, next isOn]

        def _calc_angle(a, b):
            return math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1])
        # TODO: Что-то туплю... Надо посчитать угол между прошлым вектором и текущим.
        #   Это нужно для будущего расчёта перегрузок головки, если модель её на 180 градусов разворачивает.

        self._angle = 0

        # старые координаты - (self._cur_x, self._cur_y)
        # новые координаты - (next_x, next_y)
        # Между ними проводим прямую.
        # По прямой со скоростью next_speed перемещаемся
        # Если isOn = 1 - делаем работу.
        # Создаём клон пустого листа
        # Идём в цикле по X, считаем координату на гипотенузе
        #   Считаем, сколько времени займёт перемещение в эту точку
        #   Начиная с этой точки добавляем на клон-лист матрицу с работой (умноженную на долю времени)
        # Из клон-матрицы с помощью маски вычисляем 2 матрицы:
        #  - матрица работы, которая попала на деталь,
        #  - и матрица работы, которая попала мимо детали.
        # Эти две матрицы добавляем к матрицам self._work_done и self._work_in_vain.

        self.state = (x, x_dot, theta, theta_dot)
        reward = 0
        terminated = False

        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}


    def close(self):
        pass


