
% Вычисление суммы 1/1! + 1/2! + ... + 1/10!

n = 10; % Количество членов ряда
sum_result = 0; % Инициализация суммы

% Вычисление суммы
for k = 1:n
    sum_result = sum_result + 1 / factorial(k);
end

% Вывод результата
fprintf('Сумма 1/1! + 1/2! + ... + 1/10! равна: %f\n', sum_result);

% Возвращение результата (необязательно, если не используется в других функциях)
result = sum_result;