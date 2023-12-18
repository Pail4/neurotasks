% Определение нечетких множеств A, B и C
x = 0:0.1:10; % Определение диапазона значений

% Используем встроенные функции Fuzzy Logic Toolbox для определения функций принадлежности
A = trapmf(x, [0 2 3 4]); % Трапециевидная функция принадлежности для A
B = trimf(x, [2 3 4]); % Треугольная функция принадлежности для B
C = zmf(x, [4 6]); % Z-образная функция принадлежности для C

% Построение функции принадлежности для D = A∪B∩C
D = max(min(A, B), C); % Используем максиминный метод

% Определение степени принадлежности элемента к множеству D
element = 5;
membership_degree = interp1(x, D, element, 'linear'); % Интерполяция для определения степени принадлежности

% Вывод результатов
disp(['Степень принадлежности элемента ', num2str(element), ' к множеству D: ', num2str(membership_degree)]);

% Построение графиков функций принадлежности
figure;
subplot(5,1,1); plot(x, A); title('A'); ylim([0 1]);
subplot(5,1,2); plot(x, B); title('B'); ylim([0 1]);
subplot(5,1,3); plot(x, C); title('C'); ylim([0 1]);
subplot(5,1,4); plot(x, D); title('D'); ylim([0 1]);
subplot(5,1,5); plot(x, D); hold on; plot(element, membership_degree, 'ro'); title('D с элементом'); ylim([0 1]);
