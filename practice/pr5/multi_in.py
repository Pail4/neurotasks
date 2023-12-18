import numpy as np
# Загрузить и подготовить тренировочные данные из формата CSV в список
training_data = open("train.csv", 'r') # 'r' - открываем файл для чтения
training_data_list = training_data.readlines() # readlines() - читает все строки в файле в переменную training_data_list
training_data.close() # закрываем файл csv

# Загрузить и подготовить тестовые данные из формата CSV в список
test_data = open("test.csv", 'r') # 'r' - открываем файл для чтения
test_data_list = test_data.readlines()# Загрузить и подготовить тестовые данные из формата CSV в список 
test_data.close() # закрываем файл csv
# Определение класса нейронной сети
class neuron_Net:
    
    # Инициализация весов нейронной сети
    def __init__(self, input_num, neuron_num, learningrate): #констр.(кол-во входов, кол-во нейронов)
                                         # МАТРИЦА ВЕСОВ
        # Задаем матрицу весов как случайное от -0,5 до 0,5
        self.weights = (np.random.rand(neuron_num, input_num) -0.5) 
        # Задаем параметр скорости обучения
        self.lr = learningrate
        
        pass
    
    # Метод обучения нейронной сети
    def train(self, inputs_list, targets_list): # принимает (вх. список данных, ответы)
        # Преобразовать список входов в вертикальный массив. .T - транспонирование
        inputs_x = np.array(inputs_list, ndmin=2).T # матрица числа
        targets_Y = np.array(targets_list, ndmin=2).T # матрица ответов: какое это число
        
                                           # ВЫЧИСЛЕНИЕ СИГНАЛОВ
        # Вычислить сигналы в нейронах. Взвешенная сумма.
        x = np.dot(self.weights, inputs_x) # dot - умножение матриц X = W*I = weights * inputs
        # Вычислить сигналы, выходящие из нейрона. Функция активации - сигмоида(x)
        y = 1/(1+np.exp(-x))        
        
                                            # ВЫЧИСЛЕНИЕ ОШИБКИ
        #  Ошибка E = -(цель - фактическое значение) 
        E = -(targets_Y - y) 
        
                                            # ОБНОВЛЕНИЕ ВЕСОВ
        # Меняем веса по каждой связи
        self.weights -= self.lr * np.dot((E * y * (1.0 - y)), np.transpose(inputs_x))
        
        pass
    
    # Метод прогона тестовых значений
    def query(self, inputs_list): # Принимает свой набор тестовых данных
        # Преобразовать список входов в вертикальный 2D массив. 
        inputs_x = np.array(inputs_list, ndmin=2).T 
        
        # Вычислить сигналы в нейронах. Взвешенная сумма.
        x = np.dot(self.weights, inputs_x)
        # Вычислить сигналы, выходящие из нейрона. Сигмоида(x)
        y = 1/(1+np.exp(-x))
        
        return y
                            # ЗАДАЁМ ПАРАМЕТРЫ СЕТИ
# Количество входных данных, нейронов
data_input = 15
data_neuron = 2

# Cкорость обучения
learningrate = 0.1

# Создать экземпляр нейронной сети
n = neuron_Net(data_input, data_neuron, learningrate)
                            # ОБУЧЕНИЕ
# Зададим количество эпох
epochs = 40000
# Прогон по обучающей выборке
for e in range(epochs):
    for i in training_data_list:
        # Получить входные данные числа
        all_values = i.split(',') # split(',') - раздел строку на символы где запятая "," символ разделения
        inputs_x = np.asfarray(all_values[1:])
        
        # Получить целевое значение Y, (ответ - какое это число)
        targets_Y = int(all_values[0])  # перевод символов в int, 0 элемент - ответ
        
        # создать целевые выходные значения (все 0.01, кроме нужной метки, которая равна 0.99)
        targets_Y = np.zeros(data_neuron) + 0.01
        
        # Получить целевое значение Y, (ответ - какое это число). all_values[0] - целевая метка для этой записи
        if int(all_values[0]) <= 1: # цель <= 1 потому как распознаём только 2 числа, 0 и 1.
            targets_Y[int(all_values[0])] = 0.99
            
        n.train(inputs_x, targets_Y) # наш метод train - обучение нейронной сети                      
# Вывод обученных весов
print('Весовые коэффициенты:\n', n.weights)

# Еще раз пройдем по обучающей выборке
for i in training_data_list:
    all_values = i.split(',') # split(',') - раздел строку на символы где запятая "," символ разделения
    inputs_x = np.asfarray(all_values[1:])
    # Прогон по сети
    outputs = n.query(inputs_x)
    print(i[0], 'Вероятность:\n', outputs)

# Если вероятность больше 0,5 и номер выхода совпадает с ответом, то считаем что сеть, 
#на своем определенном выходе, узнала цифру. 
for i in training_data_list:
    all_values = i.split(',') # split(',') - раздел строку на символы где запятая "," символ разделения
    inputs_x = np.asfarray(all_values[1:])
    # Прогон по сети
    outputs = n.query(inputs_x)
    # индекс самого высокого значения соответствует метке
    label = np.argmax(outputs)
    if outputs[label]>0.5 and int(all_values[0]) == label:
        print(i[0], 'Узнал?: ', 'Да!')
    else:
        print(i[0], 'Узнал?: ', 'Нет!') 

    # Проход по тестовой выборке
t = 0 # Счетчик номера нуля тестовой выборки
t1 = 0 # Счетчик номера единицы тестовой выборки
for i in test_data_list:
    all_values = i.split(',') # split(',') - раздел строку на символы где запятая "," символ разделения
    inputs_x = np.asfarray(all_values[1:])
    t += 1
    # Прогон по сети
    outputs = n.query(inputs_x)
    # индекс самого высокого значения соответствует метке
    label = np.argmax(outputs)
    if t <= 6:
        print('Вероятность что узнал 0 -',t, '?', outputs[label])
    else:
        t1 += 1
        print('Вероятность что узнал 1 -',t1, '?', outputs[label])

    t = 0 # Счетчик номера нуля тестовой выборки
t1 = 0 # Счетчик номера единицы тестовой выборки
# Если вероятность больше 0,5 и номер выхода совпадает с ответом, то считаем что сеть, 
#на своем определенном выходе, узнала цифру. 
for i in test_data_list:
    all_values = i.split(',') # split(',') - раздел строку на символы где запятая "," символ разделения
    inputs_x = np.asfarray(all_values[1:])
    # Прогон по сети
    outputs = n.query(inputs_x)
    # индекс самого высокого значения соответствует метке
    label = np.argmax(outputs)
    t += 1
    if outputs[label]>0.5 and int(all_values[0]) == label:
        print(i[0], 'Узнал?:',t, 'Да!')
    else:
        t1 += 1
        print(i[0], 'Узнал?:',t1, 'Нет!')