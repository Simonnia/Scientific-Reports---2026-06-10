fprintf('=== 生成1000个18维度TP-MLHD样本并进行可视化 ===\n');
    
    m = 1000;   
    n = 18;  
    
    tic;
    try
        
        samples = TP_MLHD_large(m, n, 400);
            
        evaluate_samples_detailed(samples);
                     visualize_samples(samples, n);
                        filename = sprintf('TP_MLHD_samples_%dx%d.mat', m, n);
        save(filename, 'samples');
        fprintf('\n样本已保存到: %s\n', filename);
       
        saveas(gcf, sprintf('TP_MLHD_visualization_%dx%d.png', m, n));
        fprintf('可视化图形已保存\n');
        
          time_elapsed = toc;
        fprintf('\n总运行时间: %.2f 秒 (%.2f 分钟)\n', time_elapsed, time_elapsed/60);
        
    catch ME
        fprintf('错误: %s\n', ME.message);
        fprintf('请检查输入参数或调整算法参数\n');
    end
function samples = TP_MLHD_large(m, n, maxIter)
    if nargin < 3
        maxIter = 400;
    end
    
    fprintf('生成 %d×%d TP-MLHD样本...\n', m, n);
    
    samples = MLHD_large_scale(m, n, maxIter);
    
    fprintf('样本生成完成\n');
end

function samples = MLHD_large_scale(m, n, maxIter)
    fprintf('优化MLHD生成 %d×%d 样本...\n', m, n);
        p = 8;     
    samples = randomLHD(m, n);
    best_samples = samples;
    best_phi = calculate_phi_fast(samples, p);
    
    T0 = 1;
    T = T0;
    coolingRate = 0.92;
    
    for iter = 1:maxIter
              col_idx = randi(n);
        row_idx = randperm(m, 2);
        
        new_samples = samples;
        temp = new_samples(row_idx(1), col_idx);
        new_samples(row_idx(1), col_idx) = new_samples(row_idx(2), col_idx);
        new_samples(row_idx(2), col_idx) = temp;
                new_phi = calculate_phi_fast(new_samples, p);
                if new_phi < best_phi
            samples = new_samples;
            best_samples = new_samples;
            best_phi = new_phi;
        else
            delta_phi = new_phi - best_phi;
            if rand() < exp(-delta_phi / T)
                samples = new_samples;
            end
        end
        
        T = T * coolingRate;
        
        if mod(iter, 50) == 0
            fprintf('迭代进度: %d/%d, φ_p: %.6f\n', iter, maxIter, best_phi);
        end
    end
    
    samples = best_samples;
end

function samples = randomLHD(m, n)
    samples = zeros(m, n);
    for i = 1:n
        samples(:, i) = (randperm(m)' - 0.5) / m;
    end
end

function phi = calculate_phi_fast(samples, p)
    [m, ~] = size(samples);
    
    if m <= 1
        phi = 0;
        return;
    end
    
    phi_sum = 0;
    batch_size = 100;  
    
    for i = 1:batch_size:m
        end_idx = min(i+batch_size-1, m);
        current_batch = i:end_idx;
        
        for j = current_batch
            if j < m
               
                dists = sqrt(sum((samples(j+1:m, :) - samples(j, :)).^2, 2));
                dists(dists == 0) = 1e-10;
                phi_sum = phi_sum + sum(dists.^(-p));
            end
        end
    end
    
    phi = phi_sum^(1/p);
end

function visualize_samples(samples, n)
    fprintf('\n开始可视化 %d×%d 样本...\n', size(samples,1), n);
    
    figure('Position', [100, 100, 1400, 1000]);
    
    subplot(2,3,1);
    plot_first_four_dimensions(samples);
    
    subplot(2,3,2);
    plot_all_dimensions_projection(samples);
    
    subplot(2,3,3);
    plot_pca_projection(samples);
    
    subplot(2,3,4);
    plot_distance_distribution(samples);
    
    subplot(2,3,5);
    plot_parallel_coordinates(samples);

    subplot(2,3,6);
    plot_space_filling_metrics(samples);
    
    sgtitle(sprintf('TP-MLHD样本可视化 (%d×%d)', size(samples,1), n), 'FontSize', 14, 'FontWeight', 'bold');
end

function plot_first_four_dimensions(samples)
    num_dims = min(4, size(samples,2));
    
    if num_dims < 2
        text(0.5, 0.5, '维度不足无法绘制', 'HorizontalAlignment', 'center');
        return;
    end
    
    selected_dims = samples(:, 1:num_dims);
    
    if num_dims == 2
        scatter(selected_dims(:,1), selected_dims(:,2), 20, 'filled', 'MarkerFaceAlpha', 0.6);
        xlabel('维度 1');
        ylabel('维度 2');
        grid on;
    else
        plotmatrix(selected_dims);
    end
    
    title('前4个维度散点图矩阵');
end

function plot_all_dimensions_projection(samples)
    [m, n] = size(samples);
    
    hold on;
    colors = parula(n);
    
    for i = 1:n
        [f, xi] = ksdensity(samples(:,i));
        plot(xi, f + (i-1)*0.3, 'Color', colors(i,:), 'LineWidth', 1.5);
        
        text(1.05, (i-1)*0.3 + 0.15, sprintf('Dim %d', i), ...
             'Color', colors(i,:), 'FontSize', 8);
    end
    
    xlabel('参数值');
    ylabel('密度分布 (偏移显示)');
    title('所有维度分布投影');
    grid on;
    xlim([0, 1]);
end

function plot_pca_projection(samples)
    [m, n] = size(samples);
    
    normalized_samples = (samples - mean(samples)) ./ std(samples);
    
    [coeff, score, ~, ~, explained] = pca(normalized_samples);
    
    scatter(score(:,1), score(:,2), 30, 'filled', 'MarkerFaceAlpha', 0.6);
    xlabel(sprintf('PC1 (%.1f%%)', explained(1)));
    ylabel(sprintf('PC2 (%.1f%%)', explained(2)));
    title('PCA降维可视化');
    grid on;
    total_variance = sum(explained(1:2));
    text(0.05, 0.95, sprintf('前2个主成分解释方差: %.1f%%', total_variance), ...
         'Units', 'normalized', 'BackgroundColor', 'white');
end

function plot_distance_distribution(samples)
    [m, n] = size(samples);
    
    num_pairs = min(5000, m*(m-1)/2);
    distances = zeros(num_pairs, 1);
    
    for k = 1:num_pairs
        i = randi(m);
        j = randi(m);
        if i ~= j
            distances(k) = norm(samples(i,:) - samples(j,:));
        end
    end
    distances = distances(distances > 0);
    
    histogram(distances, 30, 'FaceColor', [0.2, 0.6, 0.8], 'FaceAlpha', 0.7);
    xlabel('样本间欧氏距离');
    ylabel('频数');
    title('样本间距离分布');
    grid on;
    
    mean_dist = mean(distances);
    min_dist = min(distances);
    text(0.05, 0.95, sprintf('平均距离: %.3f\n最小距离: %.3f', mean_dist, min_dist), ...
         'Units', 'normalized', 'BackgroundColor', 'white');
end

function plot_parallel_coordinates(samples)

    [m, n] = size(samples);
    
    num_to_plot = min(100, m);
    plot_indices = randperm(m, num_to_plot);
    
    parallelcoords(samples(plot_indices, :), 'LineWidth', 0.5, 'Alpha', 0.3);
    xlabel('维度');
    ylabel('参数值');
    title('平行坐标图 (100个随机样本)');
    grid on;
    

    ylim([0, 1]);
end

function plot_space_filling_metrics(samples)
    [m, n] = size(samples);
    min_dist = calculate_min_distance(samples);
    phi_value = calculate_phi_fast(samples, 10);
    U_value = calculate_potential_energy(samples);
    metrics = [min_dist, phi_value, U_value/1000]; 
    metric_names = {'最小距离', 'φ_p准则', '势能/1000'};
    
    bar(metrics, 'FaceColor', [0.3, 0.7, 0.3], 'FaceAlpha', 0.7);
    set(gca, 'XTickLabel', metric_names);
    ylabel('指标值');
    title('空间填充性指标');
    grid on;
    
    for i = 1:length(metrics)
        text(i, metrics(i), sprintf('%.4f', metrics(i)), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    end
end

function min_dist = calculate_min_distance(samples)
    m = size(samples, 1);
    min_dist = inf;
        num_pairs = min(10000, m*(m-1)/2);
    for k = 1:num_pairs
        i = randi(m);
        j = randi(m);
        if i ~= j
            dist = norm(samples(i, :) - samples(j, :));
            if dist < min_dist
                min_dist = dist;
            end
        end
    end
end

function U = calculate_potential_energy(samples)
    m = size(samples, 1);
    U = 0;
    
    num_pairs = min(5000, m*(m-1)/2);
    for k = 1:num_pairs
        i = randi(m);
        j = randi(m);
        if i ~= j
            dist = norm(samples(i, :) - samples(j, :));
            U = U + 1 / (dist^2);
        end
    end
    
    U = U * (m*(m-1)/2) / num_pairs;
end

function evaluate_samples_detailed(samples)
    fprintf('\n=== 详细样本质量评估 ===\n');
    [m, n] = size(samples);
    fprintf('样本维度: %d×%d\n', m, n);
    
    fprintf('\n1. 基本统计:\n');
    fprintf('   样本范围: [%.4f, %.4f]\n', min(samples(:)), max(samples(:)));
    fprintf('   样本均值: %.4f\n', mean(samples(:)));
    fprintf('   样本标准差: %.4f\n', std(samples(:)));
    
    fprintf('\n2. 空间填充性指标:\n');
    min_dist = calculate_min_distance(samples);
    fprintf('   最小距离: %.6f\n', min_dist);
    
    phi_value = calculate_phi_fast(samples, 10);
    fprintf('   φ_p准则 (p=10): %.6f\n', phi_value);
    
    U_value = calculate_potential_energy(samples);
    fprintf('   势能准则: %.6f\n', U_value);
    
    fprintf('\n3. 投影特性检查:\n');
    check_projection_property(samples);
    fprintf('\n4. 相关性分析:\n');
    correlation_matrix = corr(samples);
    max_corr = max(abs(correlation_matrix(correlation_matrix < 0.99)));  % 排除对角线
    fprintf('   最大列间相关系数: %.4f\n', max_corr);
end

function check_projection_property(samples)
    [m, n] = size(samples);
    violations = 0;
    
    for dim = 1:n
        edges = 0:1/m:1;
        counts = histcounts(samples(:, dim), edges);
        violations = violations + sum(abs(counts - 1));
    end
    
    if violations == 0
        fprintf('   投影特性: 完美满足\n');
    else
        fprintf('   投影特性: %d 个违反\n', violations);
    end
end


