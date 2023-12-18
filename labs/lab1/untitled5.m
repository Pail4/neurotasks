% Задаем параметры
matrix_size = [16, 255];
vector_size = 16;
range_min = 5;
range_max = 10;

% Генерируем матрицу случайных чисел в заданном диапазоне
matrix = randi([range_min, range_max], matrix_size);

% Генерируем первый вектор
first_vector = -1:10:10*(vector_size-1);

% Заменяем 100-й столбец матрицы на первый вектор
matrix(:, 100) = first_vector;

% Генерируем второй вектор
second_vector = randi([range_min, range_max], 1, matrix_size(2));

% Вставляем второй вектор в 4-ю строку матрицы
matrix(4, :) = second_vector;

% Разбиваем матрицу на две равные матрицы и производим поэлементное умножение
[rows, cols] = size(matrix);
half_cols = cols / 2;

% Первая половина матрицы
matrix1 = matrix(:, 1:half_cols);

% Вторая половина матрицы
matrix2 = matrix(:, half_cols+1:end);

% Производим поэлементное умножение двух половин матрицы
result_matrix = matrix1 .* matrix2;

% Выводим часть полученной матрицы размером 4x16
disp('Часть полученной матрицы размером 4x16:');
disp(result_matrix(1:4, 1:16));