% Задаем параметры матрицы
lower_bound = -3;
upper_bound = 92;
matrix_size = [4, 4];

% Генерируем матрицу из случайных целых чисел в указанном интервале
random_matrix = randi([lower_bound, upper_bound], matrix_size);

% Выводим результат
disp('Сгенерированная матрица:');
disp(random_matrix);