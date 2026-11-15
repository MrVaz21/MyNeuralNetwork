# MyNeuralNetwork: полносвязная нейросеть с нуля на PyTorch autograd

Небольшая учебная библиотека, в которой многослойный перцептрон реализован **без `nn.Module` и `nn.Linear`**.
Веса и смещения это обычные тензоры, прямой проход написан вручную (`x @ W.T + b`),
а от PyTorch используются только автоматическое дифференцирование и оптимизаторы.

Цель проекта: разобраться, что происходит внутри сети (инициализация, активации, функции потерь,
мини-батчи, оптимизаторы), и сделать каждую из этих частей легко заменяемой.

## Возможности

- **Гибкое описание слоёв**: каждый слой задаётся кортежем `(размер, активация, инициализация)`
- **Встроенные активации**: сигмоида (по умолчанию), `ReLU`, `Linear`
- **Свои активации** через декоратор (любая функция на операциях `torch`)
- **Встроенные инициализаторы весов**: `Xavier`, `Kaiming`, `simple_random`
- **Свои инициализаторы** через декоратор, задаются отдельно для каждого слоя
- **Функции потерь**: `mse`, `cross_entropy` (индексы классов, one-hot или мягкие метки)
- **Оптимизаторы**: `SGD`, `Adam`
- **Векторизованное обучение** с перемешиваемыми **мини-батчами** (`batch_size`)
- **Заморозка** выбранных весов и смещений
- **Сохранение и загрузка весов**, автосохранение в `autosave.txt` по `Ctrl+C`
- Работает на CPU и CUDA

## Установка

```bash
git clone <ссылка-на-репозиторий>
cd <папка-репозитория>
pip install -r requirements.txt   # torch, scikit-learn, pytest
```

## Быстрый старт

Рукописные цифры (картинки 8x8, 10 классов):

```python
import random, torch
from sklearn.datasets import load_digits
from mynn.MyNeuralNetwork import MyNeuralNetwork

random.seed(0); torch.manual_seed(0)

d = load_digits()
X = torch.tensor(d.data / 16.0, dtype=torch.float32)
Y = torch.nn.functional.one_hot(torch.tensor(d.target), 10).float()

perm = torch.randperm(len(X))
X, Y = X[perm], Y[perm]
X_train, Y_train, X_test, Y_test = X[:1400], Y[:1400], X[1400:], Y[1400:]

net = MyNeuralNetwork(
    [(64,), (64, "ReLU", "Kaiming"), (10, "Linear", "Xavier")],
    neuralnetwork_name="digits",
)
net.train(200, 0.01, X_train, Y_train,
          error_type="cross_entropy", optimizer_type="Adam", batch_size=64)

pred = net.get_predict(X_test).argmax(dim=1)
print(f"Test accuracy: {(pred == Y_test.argmax(dim=1)).float().mean().item():.2%}")
```

## Описание архитектуры

```python
MyNeuralNetwork(levels_data, device="cpu", neuralnetwork_name="my_net")
```

`levels_data` это список кортежей, по одному на слой, начиная со входного:

| Кортеж | Значение |
|---|---|
| `(5,)` | входной слой на 5 признаков |
| `(16, "ReLU", "Kaiming")` | 16 нейронов, активация ReLU, веса, входящие в этот слой, инициализируются по Kaiming |
| `(1, "Linear", "Xavier")` | 1 выходной нейрон без нелинейности (регрессия или логиты) |

Если активация не указана или равна `None`, используется сигмоида. Параметр `neuralnetwork_name`
обязателен: это имя папки, куда сохраняются веса.

Как выбрать выходной слой:

| Задача | Активация выхода | Функция потерь |
|---|---|---|
| Регрессия | `"Linear"` | `mse` |
| Бинарная классификация | сигмоида (по умолчанию) | `mse` |
| Многоклассовая классификация | `"Linear"` (логиты) | `cross_entropy` |

## Свои активации

Функция пишется на операциях `torch` (не на `math`, не на встроенном `max`, без `.item()`),
чтобы autograd мог её продифференцировать:

```python
net = MyNeuralNetwork([(2,), (8, "LeakyReLU", "Kaiming"), (1,)], neuralnetwork_name="demo")

@net.add_custom_func("LeakyReLU")
def leaky_relu(z):
    return torch.maximum(z, 0.01 * z)
```

Функция должна возвращать тензор, иначе при первом вызове будет `ValueError`.
Регистрируйте её до вызова `train`.

## Свои инициализаторы весов

Функция получает `(prev_level_size, cur_level_size)` и возвращает число для одного веса:

```python
@MyNeuralNetwork.add_custom_w_init_function("LeCun")
def lecun(n_in, n_out):
    return random.gauss(0, (1 / n_in) ** 0.5)
```

Регистрируйте её **до** создания сети, которая её использует.

## Обучение

```python
net.train(
    epochs, learning_rate, inputs, outputs,
    error_type="mse",          # или "cross_entropy"
    optimizer_type="SGD",      # или "Adam"
    batch_size=None,           # None = весь датасет за шаг
    weights_to_freeze=None,    # список номеров уровней
    bies_to_freeze=None,
)
```

- Данные один раз переводятся в тензоры, а новая случайная перестановка берётся на каждой эпохе.
- Для `cross_entropy` метки могут быть one-hot (`(N, C)`) или мягкими; выходной слой должен выдавать логиты.
- `Ctrl+C` останавливает обучение и записывает `autosave.txt` в папку сети.

Предсказание: `net.get_predict(x)` принимает один пример или батч и возвращает сырой выход последнего слоя
(для классов добавьте `.argmax(dim=1)`).

## Сохранение и загрузка

```python
net.save_weights("weights.txt")
net.load_weights("weights.txt")   # архитектура должна совпадать
```

Файлы записываются в папку `networks/<neuralnetwork_name>/`.

## Примеры

```bash
python examples/digits.py        # классификация цифр: ReLU + cross-entropy + Adam
python examples/add_numbers.py   # регрессия: сеть учится складывать 5 чисел
```

## Результаты

| Задача | Конфигурация | Метрика на отложенной выборке |
|---|---|---|
| Цифры 8x8 | 64-64-10, ReLU, cross-entropy, Adam, батч 64, 200 эпох | accuracy: **97.48%** |
| Сумма 5 чисел | линейная сеть, MSE, Adam | MAE: _впишите после запуска_ |

Добавьте сюда кривую loss или график границы решений: картинка убеждает лучше таблицы.

## Тесты

```bash
pytest
```

Тесты проверяют формы весов, прямой проход на батче, сохранение и загрузку весов, а также то,
что сеть действительно обучается складывать числа.

## Структура проекта

```
mynn/
    MyNeuralNetwork.py    # класс сети
examples/
    digits.py
    add_numbers.py
tests/
    test_network.py
```

## Известные ограничения

- Только полносвязные слои (нет свёрток и рекуррентных слоёв).
- Нет регуляризации (dropout, weight decay), валидационного цикла и ранней остановки.
- Веса хранятся в текстовом файле, что медленно для больших сетей.
- Инициализация идёт по одному весу в циклах Python, поэтому очень большие слои создаются медленно.
- Смещения создаются и для входного слоя, но нигде не используются.
- Нумерация в `weights_to_freeze` и `bies_to_freeze` с единицы (уровень 1 это входной слой).
- Поддерживаются только `SGD` и `Adam`.

## Планы развития

- Слои как объекты и контейнер `Sequential` (основа для свёрточных слоёв)
- Валидационная выборка, train/val loss, ранняя остановка
- L2-регуляризация и dropout
- Сохранение через `torch.save`
- Сравнение с эквивалентной моделью на `nn.Sequential`

## Лицензия

MIT (добавьте файл `LICENSE`).