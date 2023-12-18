import numpy as np
# Загрузить и подготовить тренировочные данные из формата CSV в список
training_data = open("train.csv", 'r') # 'r' - открываем файл для чтения
training_data_list = training_data.readlines() # readlines() - читает все строки в файле в переменную training_data_list
training_data.close() # закрываем файл csv
# Определение класса нейронной сети
class neuron_Net:
    
    # Инициализация параметров нейронной сети
    def __init__(self, input_num, neuron_num, output_num, learningrate):
                                         # МАТРИЦА ВЕСОВ
        # Задаем матрицу весов как случайное
        self.weights = (np.random.rand(neuron_num, input_num) +0.0)
        self.weights_out = (np.random.rand(output_num, neuron_num) +0.0)
        
        # Задаем параметр скорости обучения
        self.lr = learningrate
        
        pass
    
    # Метод обучения нейронной сети
    def train(self, inputs_list, targets_list): # принимает (вх. список данных, ответы)
        # Преобразовать список входов в вертикальный массив. .T - транспонирование
        inputs_x = np.array(inputs_list, ndmin=2).T # матрица числа
        targets_Y = np.array(targets_list, ndmin=2).T # матрица ответов
        
                                           # ВЫЧИСЛЕНИЕ СИГНАЛОВ
        # Вычислить сигналы в нейронах скрытого слоя. Взвешенная сумма.
        x1 = np.dot(self.weights, inputs_x) # dot - умножение матриц X = W*I = weights * inputs
        # Вычислить сигналы, выходящие из нейронов скрытого слоя. Функция активации - сигмоида(x)
        y1 = 1/(1+np.exp(-x1))
        # Вычислить сигналы в нейронах выходного слоя. Взвешенная сумма.
        x2 = np.dot(self.weights_out, y1) # dot - умножение матриц X = W*I = weights * inputs
        
                                            # ВЫЧИСЛЕНИЕ ОШИБКИ
        #  Ошибка выходного слоя: E = -(цель - фактическое значение) 
        E = -(targets_Y - x2)
        # Скрытая ошибка слоя
        E_hidden = np.dot(self.weights_out.T, E) 
        
                                            # ОБНОВЛЕНИЕ ВЕСОВ
        # Меняем веса связей, исходящих из скрытого слоя
        self.weights_out -= self.lr * np.dot((E * x2), np.transpose(y1))
        # Меняем веса связей, исходящих из входного слоя
        self.weights -= self.lr * np.dot((E_hidden * y1 * (1.0 - y1)), np.transpose(inputs_x))
        
        pass
    
    # Метод прогона тестовых значений
    def query(self, inputs_list): # Принимает свой набор тестовых данных
        # Преобразовать список входов в вертикальный 2D массив. 
        inputs_x = np.array(inputs_list, ndmin=2).T 
        
        # Вычислить сигналы в нейронах скрытого слоя. Взвешенная сумма.
        x1 = np.dot(self.weights, inputs_x) # dot - умножение матриц X = W*I = weights * inputs
        # Вычислить сигналы, выходящие из нейронов скрытого слоя. Функция активации - сигмоида(x)
        y1 = 1/(1+np.exp(-x1))
        # Вычислить сигналы в нейронах выходного слоя. Взвешенная сумма.
        x2 = np.dot(self.weights_out, y1) # dot - умножение матриц X = W*I = weights * inputs
        
        return x2
                            # ЗАДАЁМ ПАРАМЕТРЫ СЕТИ
# Количество входных данных, слоев, нейронов
data_input = 2
data_neuron = 2
data_output = 1

# Cкорость обучения
learningrate = 0.2

# Создать экземпляр нейронной сети
n = neuron_Net(data_input, data_neuron, data_output, learningrate)
                            # ОБУЧЕНИЕ
# Зададим количество эпох
epochs = 70000
# Прогон по обучающей выборке
for e in range(epochs):
    for i in training_data_list:
        
        # Получить входные данные числа
        all_values = i.split(',') # split(',') - раздел строку на символы где запятая "," символ разделения
        inputs_x = np.asfarray(all_values[1:])
        
        # Получить целевое значение Y, (ответ - какое это число)
        targets_Y = int(all_values[0])  # перевод символов в int, 0 элемент - ответ
        #targets_Y = np.asfarray(all_values[0],int) 
            
        n.train(inputs_x, targets_Y) # наш метод train - обучение нейронной сети                      
# Вывод обученных весов
print('Весовые коэффициенты:\n', n.weights)

# Прогоним входные данные из обучающей выборки через обученную сеть
for i in training_data_list:
    all_values = i.split(',') # split(',') - разделить строку на символы где запятая "," символ разделения
    #all_values = np.asfarray(all_values,int) # Перевод списка в int 
    inputs_x = np.asfarray(all_values[1:])
    # Прогон по сети
    outputs = n.query(inputs_x)
    print(int(all_values[1]), 'XOR', int(all_values[2]), '=' , float(outputs), '\n')      
