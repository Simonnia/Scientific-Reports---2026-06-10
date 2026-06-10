clear; clc; close all;
N = 30;           
Dim = 18;         
max_iter = 100;   
lb = zeros(1, Dim); 
ub = ones(1, Dim);   
Gp = 0.5;         

positions = initialization(N, Dim, lb, ub);

fitness = zeros(N, 1);
for i = 1:N
    fitness(i) = objective_function(positions(i, :));
end

[best_fitness, best_index] = min(fitness);
global_best = positions(best_index, :);
global_best_fitness = best_fitness;

convergence_curve = zeros(max_iter, 1);

for T = 1:max_iter
    for i = 1:N
        if rand() > Gp
            t
            theta = (pi/2) * (T / max_iter);
            Et = ((max_iter - T) / max_iter) * cos(theta);
            
            if Dim > 5
                
                p_dimensions = randperm(Dim, 5); 
                
                for p_idx = 1:length(p_dimensions)
                    p = p_dimensions(p_idx);
                    a1 = (2*rand() - 1) * pi;
                    
                    if rand() <= 0.5
                        
                        new_position = positions(i, p) + a1 * (global_best(p) - positions(i, p)) * cos(theta);
                    else
                      
                        new_position = positions(i, p) - a1 * (global_best(p) - positions(i, p)) * sin(theta);
                    end
                                                            positions(i, p) = new_position;
                end
            else
               
                p = randi(Dim); 
                A1 = 2*rand() - 1; 
                A2 = 2*rand() - 1; 
                
                           k1 = randi(N);
                k2 = randi(N);
                while k1 == i
                    k1 = randi(N);
                end
                while k2 == i || k2 == k1
                    k2 = randi(N);
                end
                
                                new_position = Et * positions(i, p) + ...
                              A1 * (positions(k1, p) - positions(i, p)) + ...
                              A2 * (positions(k2, p) - positions(i, p));
                positions(i, p) = new_position;
            end
            
        else
                      if i ~= N
                                distances = zeros(1, 5);
                selected_starfish = randperm(N, 5);
                
                for m = 1:5
                    mp = selected_starfish(m);
                    distances(m) = norm(global_best - positions(mp, :));
                end       
                selected_distances = distances(randperm(5, 2));
                r1 = rand();
                r2 = rand();
                            new_position_vector = positions(i, :) + ...
                                    r1 * selected_distances(1) + ...
                                    r2 * selected_distances(2);
                positions(i, :) = new_position_vector;
                
            else
                                regeneration_factor = exp(-T * N / max_iter);
                positions(i, :) = regeneration_factor * positions(i, :);
            end
        end
                positions(i, :) = bound_constraint(positions(i, :), lb, ub);
        
                new_fitness = objective_function(positions(i, :));

        if new_fitness < fitness(i)
            fitness(i) = new_fitness;
        end
    end
    
    [current_best_fitness, current_best_index] = min(fitness);
    if current_best_fitness < global_best_fitness
        global_best = positions(current_best_index, :);
        global_best_fitness = current_best_fitness;
    end
    convergence_curve(T) = global_best_fitness;
    
    if mod(T, 10) == 0
        fprintf('迭代次数: %d, 最佳适应度: %.6f\n', T, global_best_fitness);
    end
end

fprintf('\n=== SFOA优化结果 ===\n');
fprintf('最佳适应度: %.6f\n', global_best_fitness);
fprintf('最佳参数: \n');
disp(global_best);
figure;
plot(1:max_iter, convergence_curve, 'r-', 'LineWidth', 2);
xlabel('迭代次数');
ylabel('最佳适应度');
title('海星优化算法(SFOA)收敛曲线');
grid on;

plot_optimized_curves(global_best);
function positions = initialization(N, Dim, lb, ub)
    positions = zeros(N, Dim);
    for i = 1:N
        positions(i, :) = lb + (ub - lb) .* rand(1, Dim);
    end
end
function X = bound_constraint(X, lb, ub)
    for j = 1:length(X)
        if X(j) < lb(j)
            X(j) = lb(j);
        elseif X(j) > ub(j)
            X(j) = ub(j);
        end
    end
end
function error = objective_function(params)
      generated_curves = proxy_model(params);
       experimental_curves = rand(7, 122); 
            total_error = 0;
    for curve_idx = 1:7
        for point_idx = 1:122
            error_val = (generated_curves(curve_idx, point_idx) - ...
                       experimental_curves(curve_idx, point_idx))^2;
            total_error = total_error + error_val;
        end
    end
    
    error = sqrt(total_error / (7 * 122));
end
function curves = proxy_model(params)
    curves = zeros(7, 122);
    for i = 1:7
              x = linspace(0, 1, 122);
               param_weight = params(mod(i-1, 6)+1 : mod(i-1, 6)+3);
        curves(i, :) = sin(2*pi*x * param_weight(1)) .* ...
                      exp(-x * param_weight(2)) + ...
                      param_weight(3) * x;
    end
end
function plot_optimized_curves(best_params)
    generated_curves = proxy_model(best_params);
            experimental_curves = rand(7, 122); % 用随机数据模拟
    
    figure;
    for i = 1:7
        subplot(3, 3, i);
        plot(1:122, experimental_curves(i, :), 'r-', 'LineWidth', 2, 'DisplayName', '实验曲线');
        hold on;
        plot(1:122, generated_curves(i, :), 'b--', 'LineWidth', 1.5, 'DisplayName', '优化曲线');
        title(['曲线 ' num2str(i)]);
        xlabel('点索引');
        ylabel('值');
        legend;
        grid on;
    end
    
    subplot(3, 3, 8);
        bar(1:18, best_params);
    title('优化后的参数值');
    xlabel('参数索引');
    ylabel('参数值');
    grid on;
    final_error = objective_function(best_params);
    subplot(3, 3, 9);
    text(0.1, 0.5, sprintf('最终误差: %.6f', final_error), 'FontSize', 12);
    axis off;
end
function rmse = calculate_rmse(predicted, actual)
    rmse = sqrt(mean((predicted - actual).^2, 'all'));
end