import copy
import math
from typing import Any, Tuple, List
import numpy as np

import gymnasium as gym
import shapely
from shapely.geometry import Point, LineString
from shapely.geometry.polygon import Polygon


class XYCutter(gym.Env):
    """
    ## Описание

    У нас в управлении есть 2D-станок с абстрактной обрабатывающей головкой.

    Все размеры задаются в миллиметрах, скорости - в м/с.

    На вход подаётся:
    - размер рабочей зоны
    - полигон с заготовкой (список с последовательными координатами углов)
    - список полигонов-отверстий внутри заготовки (их не надо обрабатывать)
    - пятно обрабатывающей головки (np.ndarray с интенсивностями обработки в каждом миллиметре) в единицу времени
    - желаемая минимальная и максимальная степень обработки


    ## Пространство действий

    Модель заточена под то, чтобы её выход было удобно переделывать в G-Code.

    Управление станком происходит путём генерации следующей точки, куда должна переместиться головка.

    Action = [next X, next Y, next Velocity, next isOn]

    next X - X-координата новой точки,
    next Y - Y-координата новой точки,
    next Velocity - скорость перемещения в новую точку (скорость влияет на интенсивность обработки),
    next isOn - 1, если головка совершает работу во время перемещения, 0 - головка выключена

    Note: Пока не учитываются импульсы, перегрузки и инерция.


    ## Пространство состояний (полностью наблюдаемая среда)

    State = [cur X, cur Y, cur Angle, cur Velocity, cur isOn, work_done_layer, work_in_vain_layer]

    cur X - X-координата текущей точки
    cur Y - Y-координата текущей точки,
    cur Velocity - скорость, с которой головка пришла в текущую точку
    cur Angle - угол в радианах, откуда пришла головка в текущую точку
    cur isOn - флаг, что головка пришла в текущую точку в рабочем состоянии.

    work_done_layer - 2D-массив с результатом сделанной работы
    work_in_vain_layer - 2D-массив с результатами напрасной работы (работа за пределами детали)

    Массивы считаются внутри функции step()


    ## Награда

    Идеальная ситуация, когда деталь обработана с заданной интенсивностью равномерно.

    Есть состояние излишней обработки. Есть состояние недостаточной обработки.

    Штрафуем за включение обрабатывающей головки над деталью (сильно штрафуем).

    За обработку не над деталью тоже штрафуем, но не сильно.


    ## Начальное состояние

    Головка расположена в координатах (0, 0)
    Скорость = 0.
    Угол = 0.
    Матрицы полезной и потерянной работы - занулены.


    ## Конец эпизода

    Эпизод заканчивается, если:

    1. Работа выполнена (выполнены условия выше)
    2. Если точек больше некоторого количества (зависит от интенсивности и требуемого уровня обработки)
    3. Если в векторе action содержатся невалидные параметры (модель должна выучить границы относительно быстро)
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
        :param desired_intensity: Желаемый уровень минимальной и максимальной работы,
          выполненной над каждой точкой детали.
        """

        self._working_area: Polygon = Polygon.from_bounds(*working_area.bounds)
        """Рабочая зона станка, задана отрезком."""
        self._object_polys: List[Polygon] = object_polys
        """Список обрабатываемых деталей."""
        self._protected_polys: List[Polygon] = protected_polys
        """Список зон, которые не нужно обрабатывать."""
        self._processor_intensity: np.ndarray = processor_intensity
        """Матрица с интенсивностью обработки детали в 1 секунду."""
        self._desired_intensity: Tuple[float, float] = desired_intensity
        """Желаемый минимальный и максимальный уровни обработки, выполненной над каждой точкой детали."""

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

        # TODO:
        #   1) вести журнал действий
        #   2) вести журнал времени на каждое действие
        #   3) вести пройденный путь на каждой действие

        self._state = []
        """Вектор текущего состояния системы"""
        self._reward: float = 0.0
        """Текущая накопленная награда/штраф среды за выполненную работу"""
        self._actions = []
        """Журнал действий"""
        self._done: int = 0
        """Успешное завершение эпизода"""
        self._terminated: int = 0
        """Завершение эпизода с ошибкой"""

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

        assert all([len(poly.boundary.xy[0]) > 2 for poly in self._object_polys]), \
            'Обрабатываемый объект должен состоять минимум из 3 углов.'

        assert all([len(poly.boundary.xy[0]) > 2 for poly in self._protected_polys]), \
            'Каждая необрабатываемая область должна состоять минимум из 3 углов.'

        assert (self._processor_intensity.shape[0] > 0) and (self._processor_intensity.shape[0] % 2 == 1), \
            'Ширина обрабатываемой зоны должна быть положительной и нечётной.'
        assert (self._processor_intensity.shape[1] > 0) and (self._processor_intensity.shape[1] % 2 == 1), \
            'Длина обрабатываемой зоны должна быть положительной и нечётной.'
        assert self._processor_intensity.min() >= 0, \
            'Внутри обрабатываемой зоны не может делаться отрицательная работа.'

        assert self._desired_intensity[0] > 0, \
            'Желаемый минимальный уровень обработки не может быть нулевым или отрицательным.'
        assert self._desired_intensity[1] > 0, \
            'Желаемый максимальный уровень обработки не может быть нулевым или отрицательным.'
        assert self._desired_intensity[0] < self._desired_intensity[1], \
            'Желаемый минимальный уровень должен быть меньше желаемого максимального уровня обработки.'

    def reset(self,
              *,
              seed: int | None = None,
              options: dict[str, Any] | None = None):
        """
        Функция переводит модель в нулевое положение.
        Откуда начинается эпизод обучения или работы.

        :param seed: Не используется (у нас пока нет случайности).
        :param options: Не используется.
        :return: Возвращает вектор состояний.
        """

        self._cur_position = Point([0, 0])
        self._cur_velocity = 0
        self._angle = 0
        self._is_on = 0

        # Матрица float32, с которой мы будем делать копии с помощью `deepcopy.copy()`
        self._dummy_nd_array = np.ndarray(shape=(int(self._working_area.bounds[2] - self._working_area.bounds[0]),
                                                 int(self._working_area.bounds[3] - self._working_area.bounds[1])),
                                          dtype=np.float32)
        self._dummy_nd_array.fill(0.0)

        # Маска из 0 и 1 в виде детали, с учётом возможных высечек.
        # С помощью неё будем определять, попала ли работа на деталь или мимо.
        self._detail_mask = np.ndarray(shape=(int(self._working_area.bounds[2] - self._working_area.bounds[0]),
                                              int(self._working_area.bounds[3] - self._working_area.bounds[1])),
                                       dtype=np.uint8)
        self._detail_mask.fill(0)

        # Заполняем маску: проверяем, что точка находится над деталью и не находится над выемкой.
        union_detail_poly = shapely.union_all(self._object_polys)
        union_protected_poly = shapely.union_all(self._protected_polys)
        detail_bounds = shapely.bounds(union_detail_poly)
        for i in range(int(self._working_area.bounds[0]), int(self._working_area.bounds[2])+1):
            for j in range(int(self._working_area.bounds[1]), int(self._working_area.bounds[3])+1):
                # Если мы над границами детали - начинаем проверять (для экономии CPU)
                if detail_bounds[0] <= i <= detail_bounds[2] and detail_bounds[1] <= j <= detail_bounds[3]:
                    point = Point(i, j)
                    chk1 = shapely.covers(union_detail_poly, point)
                    chk2 = True
                    if chk1 and self._protected_polys:
                        chk2 = not shapely.covers(union_protected_poly, point)
                    self._detail_mask[i, j] = 1 if chk1 and chk2 else 0

        # Полигон со степенью обработки каждой точки над областью детали
        self._work_done = copy.deepcopy(self._dummy_nd_array)

        # Полигон со степенью обработки каждой точки за пределами области детали (напрасная работа)
        self._work_in_vain = copy.deepcopy(self._dummy_nd_array)

        # Формируем вектор состояний (мы чуть позже изымем из него матрицы _work_done и _work_in_vain).
        self._state = [
            # граничные значения и маска детали не меняются во время обучения
            # X [cur, min, max]
            self._cur_position.coords.xy[0][0],
            self._working_area.bounds[0],
            self._working_area.bounds[2],
            # Y [cur, min, max]
            self._cur_position.coords.xy[1][0],
            self._working_area.bounds[1],
            self._working_area.bounds[3],
            # Velocity [cur, min, max]
            self._cur_velocity,
            0,
            3,  # TODO: хардкод
            # Is On [cur, min, max]
            self._is_on,
            0,
            1,  # TODO: хардкод
            # Angle [cur, min, max]
            self._angle,
            # матрицы
            self._detail_mask,
            self._work_done,
            self._work_in_vain
        ]
        self._reward = 0.0

        # Добавляем "нулевое" действие - позиционирование головки в начало координат
        self._actions = [(self._cur_position.coords.xy[0][0],
                          self._cur_position.coords.xy[1][0],
                          self._cur_velocity,
                          self._is_on)]

        return self._state

    def step(self, action: List):
        """
        Модель возвращает состояние среды после того, как агент сделал действие.

        :param action: Сделанное действие, представлено вектором значений
          [next X, next Y, next Velocity, next isOn].

        :return: Возвращает список из значений:
          [observation, reward, terminated, truncated, info]
        """
        self._actions.append(action)

        next_x = round(action[0])
        next_y = round(action[1])
        # Вычислили угол вектора (нет выколотых точек)
        next_angle = math.atan2(next_y - self._cur_position.coords.xy[1][0],
                                next_x - self._cur_position.coords.xy[0][0])
        next_velocity = action[2]  # скорость в м/с
        next_exposition = 1 / (next_velocity * 1000)  # время нахождения головки над 1 кв. мм. детали
        next_is_on = round(action[3])

        # Если входящие параметры в порядке
        if next_is_on:
            # Вычисляем гипотенузу
            distance = math.sqrt(
                (next_y - self._cur_position.coords.xy[1][0]) ** 2 +
                (next_x - self._cur_position.coords.xy[0][0]) ** 2
            )
            # Создаём массив для хранения работы, выполненной за одно действие.
            unit_of_work_array = copy.deepcopy(self._dummy_nd_array)
            # Считаем интенсивность работы с учётом времени экспозиции.
            work_to_apply = self._processor_intensity * next_exposition

            # Выполняем работу (пока без учёта границ детали)

            # Идём вдоль гипотенузы с шагом 0.1 мм
            point_processed = set()
            for dd in range(0, int(distance * 10) + 1):
                # раскладываем движение вдоль гипотенузы на движение вдоль осей
                dx_float = dd / 10 * math.cos(next_angle)
                dy_float = dd / 10 * math.sin(next_angle)
                dx = round(dx_float)
                dy = round(dy_float)
                # выполняем работу, если в этой точке мы ещё не были
                # и если точка проходит рядом с центром ячейки (0.141421 ~= 0.1 * math.sqrt(2))
                if ((dx, dy) not in point_processed
                        and abs(dx_float - dx) < 0.141421
                        and abs(dy_float - dy) < 0.141421):
                    # запоминаем точку, чтобы повторно не совершать в ней работу
                    point_processed.add((dx, dy))
                    # переходим к абсолютным координатам
                    i = int(self._cur_position.coords.xy[0][0] + dx)
                    j = int(self._cur_position.coords.xy[1][0] + dy)
                    # применяем работу
                    self._apply_work_to_point(i, j, unit_of_work_array, work_to_apply)

            # Разносим выполненную работу на 2 слоя: работа над деталью, работа за пределами детали.
            # Добавляем работу за текущее действие к соответствующим слоям.
            self._work_done += np.multiply(self._detail_mask, unit_of_work_array)
            self._work_in_vain += np.multiply((self._detail_mask * -1) + 1, unit_of_work_array)
            del unit_of_work_array, work_to_apply

        # Формируем state
        self._cur_position = Point(next_x, next_y)
        self._cur_velocity = next_velocity
        self._angle = next_angle
        self._is_on = next_is_on

        # Формируем вектор состояний (мы чуть позже изымем из него матрицы _work_done и _work_in_vain).
        self._state = [
            # граничные значения и маска детали не меняются во время обучения
            # X [cur, min, max]
            self._cur_position.coords.xy[0][0],
            self._working_area.bounds[0],
            self._working_area.bounds[2],
            # Y [cur, min, max]
            self._cur_position.coords.xy[1][0],
            self._working_area.bounds[1],
            self._working_area.bounds[3],
            # Velocity [cur, min, max]
            self._cur_velocity,
            0,
            3,  # TODO: хардкод
            # Is On [cur, min, max]
            self._is_on,
            0,
            1,  # TODO: хардкод
            # Angle [cur, min, max]
            self._angle,
            # матрицы
            self._detail_mask,
            self._work_done,
            self._work_in_vain
        ]

        self._reward, terminated, self._done = self._calc_reward()
        info = dict()

        return self._state, self._reward, terminated, self._done, info

    def _apply_work_to_point(self, i, j, unit_of_work_array, work_to_apply):
        """
        Функция применяет работу к области точек. Ничего не возвращает.

        :param i: Координата головки X.
        :param j: Координата головки Y.
        :param unit_of_work_array: Массив размером с рабочее поле, где мы накапливаем работу.
        :param work_to_apply: Матрица работы (скорость прохода головки уже учтена)
        """
        # Половина размера матрицы интенсивности (матрица интенсивности всегда нечётная по высоте и ширине)
        # Удобнее считать, что матрица с работой имеет координаты (0, 0) в самом центре.
        # Слева и снизу - отрицательные, а справа и сверху положительные (обычные оси OX и OY).
        # Матрица с работой применяется к точке (i, j) своими координатами (0, 0) и
        #   накрывает небольшую зону вокруг этой точки, если позволяют границы рабочей зоны.
        work_hw = int((self._processor_intensity.shape[0] - 1) / 2)
        work_hh = int((self._processor_intensity.shape[1] - 1) / 2)

        # Смотрим сколько надо от матрицы интенсивности отрезать слева и справа,
        # если матрица применяется вдоль границы рабочего поля.
        # work_hw = 3
        # i = 0, crop_min_x = 3    -min(0, i - work_hw)
        # i = 1, crom_min_x = 2
        # i = 2, crop_min_x = 1
        # i = 3, crop_min_x = 0
        # i = 4, crop_min_x = 0
        # max_i = 10
        # i = 6, crop_max_x = 6    work_hw + min(work_hw,  max_i - i) = 3 + min(3, 10-6) = 6
        # i = 7, crop_max_x = 6    work_hw + min(work_hw,  max_i - i) = 3 + min(3, 10-7) = 6
        # i = 8, crop_max_x = 5    work_hw + min(work_hw,  max_i - i) = 3 + min(3, 10-8) = 5
        # i = 9, crop_max_x = 4
        # i =10, crop_max_x = 3

        crop_min_x = -min(0, i - work_hw)
        crop_max_x = work_hw + min(work_hw,  int(self._working_area.bounds[2]) - i)
        crop_min_y = -min(0, j - work_hh)
        crop_max_y = work_hh + min(work_hh,  int(self._working_area.bounds[3]) - j)

        # диапазон точек, к которым применяем работу (вдоль границ что-то будет отрезаться)
        min_x = int(max(self._working_area.bounds[0], i - work_hw))
        max_x = int(min(self._working_area.bounds[2], i + work_hw))
        min_y = int(max(self._working_area.bounds[1], j - work_hh))
        max_y = int(min(self._working_area.bounds[3], j + work_hh))

        unit_of_work_array[min_y:max_y, min_x:max_x] += work_to_apply[crop_min_y: crop_max_y, crop_min_x: crop_max_x]

    def _calc_reward(self):
        """
        Функция вычисляет ценность текущего состояния и награду за текущее действие.

        :return: Возвращает 3 элемента: delta_reward, terminated, done
        """
        # Награждаем:
        # 1) Награждаем за уменьшение необработанной области (x2).
        #    Общий объём необходимой работы можно посчитать. Лучше перейти в относительные величины
        #    (сделанная / требуемая работа). Тогда максимальное вознаграждение будет 100 * 2 единиц.
        # 2) Награждаем на 0.1 за каждое действие
        #
        # Штрафуем:
        # 1) Штрафуем за работу за пределами детали (x1). Опять-таки лучше в тех же относительных величинах.
        # 2) Штрафуем за обработку детали сверх необходимого (x4).
        #    % излишне обработанных кв. мм. детали * 30 баллов
        # 3) Штрафуем за поворот головки более 90 градусов (за каждый поворот).
        #    Предварительно, на 1 единицу.
        # 4) Штрафуем за включение головки над деталью (опционально, для лазера это не нужно?)
        #    Предварительно, на 5 единиц.
        # 5) Штрафуем на 0.2 за каждое действие выше определённого количества.
        #    (короткая сторона детали мм. / 0.5 площади головки мм. * 2 действия * 2 раза (запас))
        #    (!!!) Этот штраф может помешать равномерному выполнению работы.

        reward = 0
        min_desired_level = self._detail_mask * self._desired_intensity[0]
        avg_desired_level = self._detail_mask * ((self._desired_intensity[0] + self._desired_intensity[1]) / 2)
        max_desired_level = self._detail_mask * self._desired_intensity[1]

        work_not_started = self._detail_mask * (self._work_done == 0) * 1
        work_nearly_done = (self._detail_mask
                            * self._work_done
                            * (self._work_done > 0)
                            * (self._work_done < min_desired_level))
        work_done_properly = (self._detail_mask
                              * self._work_done
                              * (self._work_done >= min_desired_level)
                              * (self._work_done <= max_desired_level))
        work_overdone = self._detail_mask * self._work_done * (self._work_done > max_desired_level)

        union_poly: Polygon = shapely.union_all(self._object_polys)
        detail_w = union_poly.bounds[2] - union_poly.bounds[0]
        detail_h = union_poly.bounds[3] - union_poly.bounds[1]
        # Средняя экспозиция - время нахождения головки над 1 кв. мм. детали TODO: (её бы в константы)
        avg_exposition = 1 / 100
        # Разница в интенсивностях обработки между желаемой и средней за один проход со скоростью 1 м/с.
        # По-хорошему, этот мультипликатор должен иметь значение в интервале [1.5, 4].
        density_diff_mult = (((self._desired_intensity[0] + self._desired_intensity[1]) / 2) /
                             (self._processor_intensity.mean() * avg_exposition))
        # минимальный размер заготовки / половина размера головки
        # * 2 прохода * разницу в интенсивностях обработки * 2 (на всякий пожарный)
        expected_num_of_passes = (min(detail_w, detail_h) / (self._processor_intensity.shape[0] - 1 / 2)
                                  * 2 * density_diff_mult * 2)

        # ======================

        # Награждаем
        # 1) Награждаем за уменьшение необработанной области (x2)
        reward += (work_nearly_done.sum() / min_desired_level.sum()) * 50
        reward += (work_done_properly.sum() / avg_desired_level.sum()) * 200
        # 2) Награждаем на 0.1 за каждое действие (нулевое действие - это начальная позиция, не учитываем)
        # reward += 0.1 * (len(self._actions) - 1)

        # Штрафуем
        # 1) Штрафуем за работу за пределами детали (x1).
        reward -= (self._work_in_vain.sum() / ((self._detail_mask * -1) + 1).sum()) * 100
        # 2) Штрафуем за обработку детали сверх необходимого (x4).
        reward -= (work_overdone.sum() / max_desired_level.sum()) * 400
        # 3) Штрафуем за поворот головки более 90 градусов (за каждый поворот).
        reward -= 1 if abs(self._angle) > math.pi / 2 else 0
        # 4) Штрафуем за включение головки над деталью (опционально, для лазера это не нужно?)
        reward -= 5 if (self._detail_mask[int(self._actions[-2][0]), int(self._actions[-2][3])] == 1
                        and round(self._actions[-2][9]) == 0
                        and round(self._actions[-1][9]) == 1) else 0
        # 5) Штрафуем на 2 за каждое действие выше определённого количества.
        #    (короткая сторона детали мм. / 0.5 площади головки мм. * 2 действия * 2 раза (запас))
        if len(self._actions) - 1 >= expected_num_of_passes:
            reward -= 2 * (len(self._actions) - 1 - expected_num_of_passes)

        # Модель (критик) интересуют последствия одного конкретного действия
        delta_reward = reward - self._reward

        # 1) Деталь обработана с нужной степенью
        done = 1 if (work_not_started.sum() == 0
                     and work_nearly_done.sum() == 0
                     and work_done_properly.sum() > 0) else 0
        # 2) Накопленная награда опустилась меньше -200
        terminated = 1 if not done and (reward < -200) else 0  # TODO: hardcode

        return delta_reward, terminated, done

    def close(self):
        pass


if __name__ == '__main__':
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

    env = XYCutter(
        working_area=LineString([[0, 0], [50, 50]]),
        object_polys=[Polygon([[10, 10], [10, 30], [30, 30], [30, 10], [10, 10]])],
        protected_polys=[Polygon([[20, 20], [20, 25], [25, 25], [25, 20], [20, 20]])],
        processor_intensity=header_intensity,
        desired_intensity=(0.008, 0.009),
    )

    action_1 = [6, 10, 1.25, 0]
    state_1 = env.step(action_1)
    action_2 = [35, 45, 1.25, 1]
    state_2 = env.step(action_2)
    action_3 = [35, 10, 1.25, 0]
    state_3 = env.step(action_3)
    action_4 = [6, 45, 0.8, 1]
    state_4 = env.step(action_4)

    print(123)
