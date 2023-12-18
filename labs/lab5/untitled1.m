% Определение переменных
x = -3:0.1:3;
y = -3:0.1:3;

% Создание сетки
[X,Y] = meshgrid(x,y);

% Определение функции
Z = (X - Y).*Y + 1;

% Ограничение Z по заданным границам
Z(Z < -4) = -4;
Z(Z > 3) = 3;

% Построение 3D графика
surf(X,Y,Z)
title('3D plot of the function z = (x-y)*y+1')
xlabel('x')
ylabel('y')
zlabel('z')
