## XY-cutter

### Описание

Это учебный проект по управлению 2D ЧПУ станком с помощью RL нейросети.  
Станок под управлением модели должен автоматически обработать деталь, если известны её размеры, местоположение и желаемый уровень обработки.  
Создан в качестве проектной работы к курсу OTUS "Обучение с подкреплением" (https://otus.ru/lessons/reinforcement-learning-cours/). 

В качестве обрабатывающей головки предполагается пока абстрактная головка (лазер, гравировка, покраска, полировка).
Для каждой конкретной реализации могут быть тонкости в функции награды.

Написана окружающая среда на базе [gymnasium](https://gymnasium.farama.org/), 
которая отслеживает выполнение работы над деталью и за её пределами.  
Работа выполняется в небольшой зоне с интенсивностью, которая задана с помощью матрицы.

### Пространство действий

Пространство действий непрерывное. Вектор действий состоит из 4 элементов: 
`[next X, next Y, next Velocity, next Is_On]`.
Это координаты следующей точки, куда станок должен переместить обрабатывающую головку.
Такой управляющий вектор будет удобно переделать в [G-Code](https://ru.wikipedia.org/wiki/G-code) 
и (после некоторой доработки) передать в [плату управления](https://mesaus.com/product/7i95t/) ЧПУ станком.

### Пространство состояний

Окружающая среда возвращает вектор из 13 цифр и 3 матриц.
В векторе с цифрами описано: 
* XY координаты головки
* скорость,
* угол перемещения (относительно предыдущего движения),
* включена ли головка или нет.
* И мин-макс допустимые значения для этих параметров (константы).

3 матрицы это:
* матрица с маской обрабатываемой детали (константа)
* матрица полезной работы
* матрица "напрасной" работы

Для обучения модели используется алгоритм DDPG с двумя критиками (быстрый и медленный) и clip'ом значений полученного action'а. (честно стырен из 10-й лекции, "DDPG Cartpole")

Архитектура модели - один конволюционный слой и 3 линейных слоя для управления станком.

### Проблемы

НЕ УЧИТСЯ. Нет роста средней награды.

Очень быстро скатывается в какой-то локальный минимум или в пределы модели. 
Но картинки выдаёт интересные, особенно по началу, когда шума в модели много. 
Хотя у модели бывают иногда просветления.

Проблемы могут быть в:
* функции награды;
* глубине или структуре сети (кажется, что 1 слой Conv2d - это маловато, попробую поглубже сделать);
* способе объединения видео-сигнала и управляющего сигнала (можно глянуть в сторону encoder-decoder или U-net, и подмешивать управляющее воздействие в серединку);
* реализации процесса обучения (подумываю позаимствовать процесс обучения у OpenAi из мануала, см. ссылки ниже); 
* том, что в модель передаются константы. Замысел был в том, чтобы модель научилась сравнивать их с соседними изменяющимися значениями, и выучила, когда окружающая среда начнёт делать terminate (пока безуспешно). 

### Что можно позапускать

1) Можно запустить просто модель окружающей среды. Там делается 4 действия и рисуется график перемещения головки.  
   `python3 environment.py`

2) Можно запустить цикл обучения. Каждую 2-ю эпоху выводится график перемещения головки. 
   Только не уверен, что это можно из консоли. Но точно можно из PyCharm.  
   `python3 train.py`

### Софт и железо

Разработка велась на PyCharm Pro, под Ubuntu 20.04.

Python 3.10.12.
Окружение ставится с помощью [PipEnv](https://docs.pipenv.org/).   
На всякий пожарный, положил рядом `requirements.txt`, но он сгенерён силами `pipenv`, и я не ручаюсь за его корректность.

Jupyter не очень люблю. 
Он функции кеширует и не перегружает их, если они изменились в пакетах, даже если сделать `del` и `import` заново.   
Ну и отлаживать в нём не удобно.

Тренировал на домашней GeForce GTX 1080. 


## Ссылки, которые помогли:

* https://spinningup.openai.com/en/latest/spinningup/rl_intro.html
* https://spinningup.openai.com/en/latest/algorithms/ddpg.html
* https://github.com/sshish/RL-DDPG/blob/master/main.ipynb

---
* https://github.com/Lornatang/ResNet-PyTorch/blob/main/model.py
* https://neurohive.io/ru/vidy-nejrosetej/resnet-34-50-101/
* https://education.yandex.ru/handbook/ml
* https://shapely.readthedocs.io/en/stable/reference/shapely.Polygon.html
* https://python-graph-gallery.com/2d-density-plot/
