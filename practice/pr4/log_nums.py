import numpy as np

# Загрузить и подготовить тренировочные данные из формата CSV в список
training_data = open("./train.csv", "r")  # 'r' - открываем файл для чтения
training_data_list = (
    training_data.readlines()
)  # readlines() - читает все строки в файле в переменную training_data_list
training_data.close()  # закрываем файл csv

# Загрузить и подготовить тестовые данные из формата CSV в список
test_data = open("./test.csv", "r")  # 'r' - открываем файл для чтения
test_data_list = (
    test_data.readlines()
)  # Загрузить и подготовить тестовые данные из формата CSV в список
test_data.close()  # закрываем файл csv
# Инициализация весов нейрона
weights = np.zeros(15)

# Скорость обучения
lr = 1

# Зададим количество эпох
epochs = 1000

# Зададим порог единичной функции активации
bias = 3
# Прогон по обучающей выборке
for e in range(epochs):
    for i in training_data_list:
        # Получить входные данные числа
        all_values = i.split(
            ","
        )  # split(',') - раздел строку на символы где запятая "," символ разделения
        inputs_x = np.asfarray(all_values[1:])

        # Получить целевое значение Y, (ответ - какое это число)
        target_Y = int(all_values[0])  # перевод символов в int, 0 элемент - ответ

        # Переводим целевой результат в бинарный вид. Так как мы ищем только значение ноль, значит только он будет верным = 1.
        # остальные ответы, будут неверными, соответственно они обращаются в ноль.
        if target_Y == 0:
            target_Y = 1
        else:
            target_Y = 0

        # Взвешенная сумма
        y = np.sum(weights * inputs_x)

        if y >= bias:
            # Когда равно или превышено пороговое значение, выход должен быть - y = 1
            y = 1

            # Ошибка E = -(целевое значение - выход нейрона)
            E = -(target_Y - y)

            # Меняем веса по каждому из входов (дельта правило)
            weights -= lr * E * inputs_x

        else:
            # Когда не превышено пороговое значение, выход должен быть - y = 0
            y = 0

            # Ошибка E = -(целевое значение - выход нейрона)
            E = -(target_Y - y)

            # Меняем веса по каждому из входов (дельта правило)
            weights -= lr * E * inputs_x
# Вывод обученых весов
print("Весовые коэффициенты:\n", weights)

# Еще раз пройдем по обучающей выборке
for i in training_data_list:
    all_values = i.split(
        ","
    )  # split(',') - раздел строку на символы где запятая "," символ разделения
    inputs_x = np.asfarray(all_values[1:])
    print(i[0], " это 0? ", np.sum(weights * inputs_x) >= bias)

# Проход по тестовой выборке
t = 0  # Счетчик номера нуля тестовой выборки
for i in test_data_list:
    all_values = i.split(
        ","
    )  # split(',') - раздел строку на символы где запятая "," символ разделения
    inputs_x = np.asfarray(all_values[1:])
    t += 1
    print("Узнал 0 - ", t, "?", np.sum(weights * inputs_x) >= bias)
