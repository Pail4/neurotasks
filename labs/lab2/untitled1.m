% Определение функций принадлежности для множеств A, B и C
X = 0:0.01:5; % Диапазон значений
mu_A = double(X <= 3); % Функция принадлежности для A
mu_B = max(0, 1 - abs((X - 3)/2)); % Функция принадлежности для B
mu_C = max(0, 1 - X/5); % Функция принадлежности для C

% Определение функции принадлежности для D = A∪B∪C
mu_D = max([mu_A; mu_B; mu_C]);

% Построение графиков функций принадлежности
figure;
plot(X, mu_A, 'r', 'LineWidth', 2); hold on;
plot(X, mu_B, 'g', 'LineWidth', 2);
plot(X, mu_C, 'b', 'LineWidth', 2);
plot(X, mu_D, 'k', 'LineWidth', 2);
legend('A', 'B', 'C', 'D');
xlabel('X');
ylabel('Степень принадлежности');
title('Функции принадлежности множеств A, B, C и D');
grid on;
hold off;
