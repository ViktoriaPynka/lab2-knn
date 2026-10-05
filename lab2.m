%% Лабораторна робота №2
% Побудова регресора методом k-найближчих сусідів (k-NN)

clear;
clc;
close all;

%% 1. Генерація набору даних
rng(15);              % фіксуємо генератор випадкових чисел
n = 1000;             % кількість значень

X = 10 * rand(n,1);   % випадкові значення X
Y = 2*X + sin(X) + randn(n,1);  % залежна змінна Y з випадковим шумом

% Графік початкових даних

figure;
scatter(X, Y, 15, 'filled');

xlabel('X');
ylabel('Y');
title('Початковий набір даних');
grid on;

%% 2. Нормалізація даних

X_norm = normalize(X, 'range');
Y_norm = normalize(Y, 'range');

% Виведення діапазону після нормалізації
fprintf('X після нормалізації: %.2f - %.2f\n', min(X_norm), max(X_norm));
fprintf('Y після нормалізації: %.2f - %.2f\n', min(Y_norm), max(Y_norm));

%% 3. Поділ даних на навчальну і тестову вибірки

cv = cvpartition(n, 'HoldOut', 0.2);

trainIdx = training(cv);
testIdx = test(cv);

X_train = X_norm(trainIdx);
Y_train = Y_norm(trainIdx);

X_test = X_norm(testIdx);
Y_test = Y_norm(testIdx);

fprintf('Навчальна вибірка: %d значень\n', length(X_train));
fprintf('Тестова вибірка: %d значень\n', length(X_test));

%% 4. Візуалізація навчальної та тестової вибірок

figure;

scatter(X_train, Y_train, 15, 'b', 'filled');
hold on;

scatter(X_test, Y_test, 15, 'r', 'filled');

xlabel('Нормалізований X');
ylabel('Нормалізований Y');
title('Поділ даних на навчальну та тестову вибірки');

legend('Навчальна вибірка', 'Тестова вибірка');
grid on;
hold off;

%% 5. KNN-регресія для різних значень K

K_values = [1 3 5 7 9 11 15 20];
MSE_values = zeros(size(K_values));

for i = 1:length(K_values)

    K = K_values(i);
    Y_pred = zeros(size(X_test));

    % Прогноз для кожного тестового значення
    for j = 1:length(X_test)

        % Відстань від тестової точки до навчальних
        distances = abs(X_train - X_test(j));

        % Сортуємо відстані
        [~, idx] = sort(distances);

        % Беремо K найближчих сусідів
        neighbors = Y_train(idx(1:K));

        % Прогноз - середнє значення сусідів
        Y_pred(j) = mean(neighbors);
    end

    % Середньоквадратична похибка
    MSE_values(i) = mean((Y_test - Y_pred).^2);

    fprintf('K = %d, MSE = %.6f\n', K, MSE_values(i));
end

%% 6. Графік залежності MSE від K

figure;

plot(K_values, MSE_values, '-o', ...
    'LineWidth', 2, 'MarkerSize', 7);

xlabel('Кількість сусідів K');
ylabel('MSE');
title('Залежність похибки MSE від K');

grid on;

%% 7. Вибір найкращого K

[minMSE, bestIndex] = min(MSE_values);
bestK = K_values(bestIndex);

fprintf('\nНайкраще K = %d\n', bestK);
fprintf('Мінімальна MSE = %.6f\n', minMSE);

%% Прогноз KNN для найкращого K

Y_best = zeros(size(X_test));

for j = 1:length(X_test)

    distances = abs(X_train - X_test(j));
    [~, idx] = sort(distances);

    neighbors = Y_train(idx(1:bestK));
    Y_best(j) = mean(neighbors);
end

%% 8. Візуалізація результату KNN-регресії

figure;

% Реальні тестові значення
scatter(X_test, Y_test, 20, 'b', 'filled');
hold on;

% Сортуємо X для правильної побудови лінії прогнозу
[X_sorted, sortIdx] = sort(X_test);
Y_sorted_pred = Y_best(sortIdx);

% Прогноз KNN
plot(X_sorted, Y_sorted_pred, 'r-', 'LineWidth', 2);

xlabel('Нормалізований X');
ylabel('Нормалізований Y');
title(['KNN-регресія при K = ', num2str(bestK)]);

legend('Реальні тестові значення', 'Прогноз KNN');
grid on;
hold off;

%% 9. Збереження результатів

save('lab2_results.mat', ...
    'X', 'Y', ...
    'X_train', 'Y_train', ...
    'X_test', 'Y_test', ...
    'K_values', 'MSE_values', ...
    'bestK', 'minMSE', 'Y_best');

fprintf('\nРезультати збережено у файл lab2_results.mat\n');