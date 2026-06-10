function AOO_optimization_curve_fitting()
       clc;
    clear;
    close all;
        N = 30;                  
    Max_iter = 100;       
    dim = 18;                 
    lb = zeros(1, dim);       
    ub = ones(1, dim);        
    
    X = initialization(N, dim, ub, lb);

    fitness = zeros(1, N);
    for i = 1:N
        fitness(i) = objective_function(X(i, :));
    end
    
    [best_fitness, best_index] = min(fitness);
    X_best = X(best_index, :);
    convergence_curve = zeros(1, Max_iter);
    
    for t = 1:Max_iter

        c = 1 - (t / Max_iter)^3;  
        
        for i = 1:N

            r = rand();
            m = 0.5 * r / dim; 
            L = N * r / dim;     
            e = 0.5 * r / dim;  
            
            
            if rand() > 0.5
              
                W = (c / pi) * (2 * rand(1, dim) - 1) .* ub;
                                if mod(i, round(N/10)) == 0
                                      X_new = mean(X, 1) + W;
                elseif mod(i, round(N/10)) == 1
                                   X_new = X_best + W;
                else
                                 X_new = X(i, :) + W;
                end
            else
                              if rand() > 0.5
                                        A = ub - abs(ub .* t .* sin(2 * pi * rand()) / Max_iter);
                    R = (m * e + L^2) * (2 * A .* rand(1, dim) - A) / dim;
                    
                                        beta = 1.5;
                    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / ...
                           (gamma((1 + beta) / 2) * beta * 2^((beta - 1)/2)))^(1/beta);
                    u = randn(1, dim) * sigma;
                    v = randn(1, dim);
                    step = u ./ abs(v).^(1/beta);
                    Levy = 0.01 * step;
                    
                    X_new = X_best + R + c * Levy .* X_best;
                else
                             B = ub - abs(ub .* t .* cos(2 * pi * rand()) / Max_iter);
                    
                    k = 0.5 + 0.5 * rand();      
                    x_param = 3 * rand() / dim; 
                    theta = pi * rand();         
                    a = (1 / pi) * exp(rand());  
                    
                    J = (2 * k * x_param^2 * sin(2 * theta)) / (m * 9.8) * ...
                        (2 * B .* rand(1, dim) - B) / dim * (1 - a);
                    
                                       beta = 1.5;
                    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / ...
                           (gamma((1 + beta) / 2) * beta * 2^((beta - 1)/2)))^(1/beta);
                    u = randn(1, dim) * sigma;
                    v = randn(1, dim);
                    step = u ./ abs(v).^(1/beta);
                    Levy = 0.01 * step;
                    
                    X_new = X_best + J + c * Levy .* X_best;
                end
            end
            
            X_new = max(X_new, lb);
            X_new = min(X_new, ub);
            
                     f_new = objective_function(X_new);
            

            if f_new < fitness(i)
                X(i, :) = X_new;
                fitness(i) = f_new;
                

                if f_new < best_fitness
                    best_fitness = f_new;
                    X_best = X_new;
                end
            end
        end
        
            convergence_curve(t) = best_fitness;
        

        if mod(t, 10) == 0
            fprintf('迭代次数: %d, 最佳适应度: %.6f\n', t, best_fitness);
        end
    end
    

    fprintf('\n优化完成!\n');
    fprintf('最佳适应度值: %.6f\n', best_fitness);
    fprintf('最佳参数: \n');
    disp(X_best);

    figure;
    plot(convergence_curve, 'LineWidth', 2);
    xlabel('迭代次数');
    ylabel('适应度值');
    title('AOO算法收敛曲线');
    grid on;
    
    plot_optimal_curves(X_best);
end

function X = initialization(N, dim, ub, lb)
    X = zeros(N, dim);
    for i = 1:N
        X(i, :) = lb + (ub - lb) .* rand(1, dim);
    end
end
function f = objective_function(x)
    generated_curves = surrogate_model(x);
    
        experimental_curves = load_experimental_curves();
    
    error = 0;
    for i = 1:7
        curve_error = sqrt(mean((generated_curves(i, :) - experimental_curves(i, :)).^2));
        error = error + curve_error;
    end
    
    f = error / 7; 
end


function curves = surrogate_model(params)
 
    x_points = linspace(0, 10, 122);
    curves = zeros(7, 122);
    
    for i = 1:7
    
        amplitude = params(mod(i-1, 6) + 1);
        frequency = params(mod(i, 6) + 7);
        phase = params(mod(i+1, 6) + 13);
        
        curves(i, :) = amplitude * sin(frequency * x_points + phase);
    end
end

function exp_curves = load_experimental_curves()
        x_points = linspace(0, 10, 122);
    exp_curves = zeros(7, 122);
    
    for i = 1:7
      
        exp_curves(i, :) = (1 + 0.1*i) * sin((0.5 + 0.1*i) * x_points + 0.2*i);
    end
end

function plot_optimal_curves(best_params)
    generated_curves = surrogate_model(best_params);
    experimental_curves = load_experimental_curves();
    x_points = linspace(0, 10, 122);
    
    figure;
    for i = 1:7
        subplot(3, 3, i);
        plot(x_points, generated_curves(i, :), 'b-', 'LineWidth', 2);
        hold on;
        plot(x_points, experimental_curves(i, :), 'r--', 'LineWidth', 2);
        title(['曲线 ' num2str(i)]);
        legend('优化曲线', '实验曲线', 'Location', 'best');
        grid on;
    end
    
    total_error = 0;
    for i = 1:7
        error_i = sqrt(mean((generated_curves(i, :) - experimental_curves(i, :)).^2));
        total_error = total_error + error_i;
        fprintf('曲线 %d 的RMSE: %.6f\n', i, error_i);
    end
    fprintf('平均RMSE: %.6f\n', total_error/7);
end