% Определение переменных
x = -3:0.1:3;
y = -3:0.1:3;
[X,Y] = meshgrid(x,y);

% Функция
Z = (X - Y).*Y + 1;

% Ограничение Z
Z(Z < -4) = -4;
Z(Z > 3) = 3;

% Создание нечеткой системы
fis = mamfis('Name',"mamdani_system");

% Добавление переменных
fis = addInput(fis,[-3 3],'Name',"x");
fis = addInput(fis,[-3 3],'Name',"y");
fis = addOutput(fis,[-4 3],'Name',"z");

% Добавление функций принадлежности к входным переменным
fis = addMF(fis,"x","trimf",[-3 0 3],'Name',"mf1");
fis = addMF(fis,"y","trimf",[-3 0 3],'Name',"mf1");

% Добавление функции принадлежности к выходной переменной
fis = addMF(fis,"z","trimf",[-4 0 3],'Name',"mf1");

% Добавление простого правила
rule = "If x is mf1 and y is mf1 then z is mf1";
fis = addRule(fis,rule);

% Построение системы
plotfis(fis);
