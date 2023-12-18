% Определение лингвистических переменных
fis = newfis('fis');

fis = addvar(fis, 'input', 'Spice Amount', [0 10]); % Количество специй
fis = addvar(fis, 'input', 'Spice Hotness', [0 10]); % Острота специй
fis = addvar(fis, 'output', 'Dish Volume', [0 10]); % Объем блюда

% Определение функций принадлежности для каждой переменной
fis = addmf(fis, 'input', 1, 'low', 'gaussmf', [1.5 0]);
fis = addmf(fis, 'input', 1, 'medium', 'gaussmf', [1.5 5]);
fis = addmf(fis, 'input', 1, 'high', 'gaussmf', [1.5 10]);

fis = addmf(fis, 'input', 2, 'low', 'gaussmf', [1.5 0]);
fis = addmf(fis, 'input', 2, 'medium', 'gaussmf', [1.5 5]);
fis = addmf(fis, 'input', 2, 'high', 'gaussmf', [1.5 10]);

fis = addmf(fis, 'output', 1, 'small', 'gaussmf', [1.5 0]);
fis = addmf(fis, 'output', 1, 'medium', 'gaussmf', [1.5 5]);
fis = addmf(fis, 'output', 1, 'large', 'gaussmf', [1.5 10]);

% Создание правил нечеткого вывода
rule1 = [1 1 1 1 1];
rule2 = [2 2 2 1 1];
rule3 = [3 3 3 1 1];

fis = addrule(fis, [rule1; rule2; rule3]);

% Генерация случайных значений для количества и остроты специй
random_spice_amount = rand * 10;
random_spice_hotness = rand * 10;

% Проведение нечеткого вывода для случайных значений
output = evalfis(fis, [random_spice_amount, random_spice_hotness]);

% Вывод случайных значений и результата
disp(['Random Spice Amount: ', num2str(random_spice_amount)]);
disp(['Random Spice Hotness: ', num2str(random_spice_hotness)]);
disp(['Output Dish Volume: ', num2str(output)]);

% Вывод графика функций принадлежности
figure;
subplot(2, 1, 1);
plotmf(fis, 'input', 1);
title('Spice Amount Membership Functions');
subplot(2, 1, 2);
plotmf(fis, 'input', 2);
title('Spice Hotness Membership Functions');

% Вывод графика нечеткого вывода
figure;
plotfis(fis);
