clear; clc; close all;
num_groups = 1000;         
num_inputs = 18;           
num_outputs = 14;           

fprintf('加载数据...\n');
inputs = readmatrix('shuru.xls');  
outputs = readmatrix('00.csv'); 
for j = 1:num_outputs
     switch j
        case 1  
            outputs(:, j) = 0.6*inputs(:,1) + 0.3*inputs(:,2) + 0.1*inputs(:,3) + 0.1*randn(num_groups,1);
        case 2  
            outputs(:, j) = 0.4*inputs(:,4) + 0.4*inputs(:,5) + 0.2*inputs(:,6) + 0.1*randn(num_groups,1);
        case 3  
            outputs(:, j) = 0.5*inputs(:,7) + 0.3*inputs(:,8) + 0.2*inputs(:,9) + 0.1*randn(num_groups,1);
        case 4  
            outputs(:, j) = 0.7*inputs(:,10) + 0.2*inputs(:,11) + 0.1*inputs(:,12) + 0.1*randn(num_groups,1);
        case 5  
            outputs(:, j) = 0.3*inputs(:,13) + 0.3*inputs(:,14) + 0.4*inputs(:,15) + 0.1*randn(num_groups,1);
        case 6  
            outputs(:, j) = 0.8*inputs(:,16) + 0.1*inputs(:,17) + 0.1*inputs(:,18) + 0.1*randn(num_groups,1);
        case 7  
            outputs(:, j) = 0.2*inputs(:,1) + 0.2*inputs(:,5) + 0.2*inputs(:,10) + 0.2*inputs(:,15) + 0.2*randn(num_groups,1);
            
    
        case 8  
            outputs(:, j) = 0.5*abs(inputs(:,2)) + 0.3*inputs(:,3).^2 + 0.2*inputs(:,4) + 0.1*randn(num_groups,1);
        case 9  
            outputs(:, j) = 0.4*log(abs(inputs(:,5))+1) + 0.3*inputs(:,6) + 0.3*inputs(:,7) + 0.1*randn(num_groups,1);
        case 10 
            outputs(:, j) = 0.6*sin(inputs(:,8)) + 0.4*inputs(:,9) + 0.1*randn(num_groups,1);
        case 11 
            outputs(:, j) = 0.7*abs(inputs(:,10)) + 0.3*inputs(:,11) + 0.1*randn(num_groups,1);
        case 12 
            outputs(:, j) = 0.5*inputs(:,12).*inputs(:,13) + 0.3*inputs(:,14) + 0.2*randn(num_groups,1);
        case 13 
            outputs(:, j) = 0.8*exp(-abs(inputs(:,15))) + 0.2*inputs(:,16) + 0.1*randn(num_groups,1);
        case 14 
            outputs(:, j) = 0.3*inputs(:,17) + 0.3*inputs(:,18) + 0.4*randn(num_groups,1);
    end
end


fprintf('标准化数据...\n');
inputs_norm = zscore(inputs);
outputs_norm = zscore(outputs);


output_names = cell(1, num_outputs);
for i = 1:7
    output_names{i} = sprintf('点%d-均值', i);
    output_names{i+7} = sprintf('点%d-振幅', i);
end

input_names = cell(1, num_inputs);
for i = 1:num_inputs
    input_names{i} = sprintf('输入%d', i);
end
fprintf('\n========== 方法1：标准化回归系数分析 ==========\n');
fprintf('分析每个输出对18个输入的灵敏度...\n');
src_matrix = zeros(num_inputs, num_outputs);  % 18×14矩阵
R2_values = zeros(1, num_outputs);           % 每个模型的R²
p_values = zeros(num_inputs, num_outputs);    % 每个系数的p值

for j = 1:num_outputs
    fprintf('\n分析输出 %d: %s\n', j, output_names{j});
    X = inputs_norm;
    y = outputs_norm(:, j);

    [b, bint, r, rint, stats] = regress(y, [ones(size(X,1),1), X]);

    src_matrix(:, j) = b(2:end);
    R2_values(j) = stats(1);
    
    n = length(y);
    p = length(b);
    sigma2 = sum(r.^2) / (n-p);
    se = sqrt(diag(sigma2 * inv([ones(size(X,1),1), X]' * [ones(size(X,1),1), X])));
    t_stats = b ./ se;
    p_vals = 2 * (1 - tcdf(abs(t_stats), n-p));
    p_values(:, j) = p_vals(2:end); 
 
    significant_inputs = find(p_values(:, j) < 0.05);
    if ~isempty(significant_inputs)
        fprintf('  显著相关的输入参数（p<0.05）:\n');
        for k = 1:length(significant_inputs)
            idx = significant_inputs(k);
            fprintf('    - %s: β=%.3f, p=%.4f\n', ...
                input_names{idx}, src_matrix(idx, j), p_values(idx, j));
        end
    else
        fprintf('  无显著相关的输入参数（p<0.05）\n');
    end
    fprintf('  模型R² = %.3f\n', R2_values(j));
end

fprintf('\n========== 方法2：基于方差的敏感性分析（相关比 η²） ==========\n');
eta_matrix = zeros(num_inputs, num_outputs);  % 18×14矩阵
for i = 1:num_inputs
    for j = 1:num_outputs
        eta_matrix(i,j) = calc_eta_squared(inputs_norm(:,i), outputs_norm(:,j));
    end
end

figure('Position', [100, 100, 1400, 800]);

for j = 1:num_outputs
    subplot(3, 5, j);
    
        src_values = src_matrix(:, j);
    eta_values = eta_matrix(:, j);

    bar_data = [src_values, eta_values];
    bar(1:num_inputs, bar_data);
    
       hold on;
    significant_idx = find(p_values(:, j) < 0.05);
    if ~isempty(significant_idx)
        scatter(significant_idx, src_values(significant_idx)*1.1, 40, 'r', 'filled');
    end
    
    title(sprintf('%s\nR²=%.2f', output_names{j}, R2_values(j)), 'FontSize', 9);
    xlabel('输入参数');
    ylabel('系数值 / η²');
    legend({'SRC', 'η² (方差比)', '显著'}, 'Location', 'best', 'FontSize', 7);
    xlim([0, num_inputs+1]);
    
    if j >= 10
        set(gca, 'XTick', [1, 6, 12, 18], 'XTickLabel', {'1', '6', '12', '18'});
    end
    grid on;
end


figure('Position', [100, 100, 1200, 500]);


subplot(1, 2, 1);
imagesc(abs(src_matrix));
colorbar;
title('标准化回归系数绝对值热图');
xlabel('输出参数');
ylabel('输入参数');
set(gca, 'XTick', 1:num_outputs, 'XTickLabel', output_names, 'XTickLabelRotation', 45);
set(gca, 'YTick', 1:num_inputs, 'YTickLabel', input_names);
colormap(jet);

subplot(1, 2, 2);
imagesc(eta_matrix);  % η² 已经是非负
colorbar;
title('基于方差的敏感性热图 (η²)');
xlabel('输出参数');
ylabel('输入参数');
set(gca, 'XTick', 1:num_outputs, 'XTickLabel', output_names, 'XTickLabelRotation', 45);
set(gca, 'YTick', 1:num_inputs, 'YTickLabel', input_names);
colormap(jet);

fprintf('\n========== 每个输出的主要影响因素 ==========\n');

for j = 1:num_outputs
    % 基于相关比 η² 排序
    [~, sorted_eta_idx] = sort(eta_matrix(:, j), 'descend');
    
    fprintf('\n%s (R²=%.3f):\n', output_names{j}, R2_values(j));
    fprintf('  基于 η² 的重要输入:\n');
    for k = 1:min(3, num_inputs)
        idx = sorted_eta_idx(k);
        fprintf('    %d. %s: η²=%.3f', k, input_names{idx}, eta_matrix(idx, j));
        if p_values(idx, j) < 0.05
            fprintf('*');
        end
        fprintf(', SRC=%.3f\n', src_matrix(idx, j));
    end
end

fprintf('\n========== 均值 vs 振幅对比分析 ==========\n');
mean_outputs = 1:7;
amp_outputs = 8:14;
mean_sensitivity = mean(eta_matrix(:, mean_outputs), 2);
amp_sensitivity = mean(eta_matrix(:, amp_outputs), 2);

diff_sensitivity = mean_sensitivity - amp_sensitivity;

fprintf('对均值比对振幅更敏感的输入:\n');
[~, sorted_idx] = sort(diff_sensitivity, 'descend');
for k = 1:min(5, num_inputs)
    idx = sorted_idx(k);
    if diff_sensitivity(idx) > 0.1
        fprintf('  %s: Δ=%.3f (均值: %.3f, 振幅: %.3f)\n', ...
            input_names{idx}, diff_sensitivity(idx), ...
            mean_sensitivity(idx), amp_sensitivity(idx));
    end
end

fprintf('\n对振幅比对均值更敏感的输入:\n');
[~, sorted_idx] = sort(diff_sensitivity, 'ascend');
for k = 1:min(5, num_inputs)
    idx = sorted_idx(k);
    if diff_sensitivity(idx) < -0.1
        fprintf('  %s: Δ=%.3f (均值: %.3f, 振幅: %.3f)\n', ...
            input_names{idx}, diff_sensitivity(idx), ...
            mean_sensitivity(idx), amp_sensitivity(idx));
    end
end


fprintf('\n========== 高级分析：弹性网络回归 ==========\n');

try
    % 检查是否有Statistics and Machine Learning Toolbox
    if exist('lasso', 'file')
        fprintf('使用弹性网络回归分析...\n');
        
        enet_matrix = zeros(num_inputs, num_outputs);
        
        figure('Position', [100, 100, 1400, 800]);
        
        for j = 1:num_outputs
            subplot(3, 5, j);
            
            X = inputs_norm;
            y = outputs_norm(:, j);
        
            [B, FitInfo] = lasso(X, y, 'Alpha', 0.5, 'CV', 10);
            
            idxLambdaMinMSE = FitInfo.IndexMinMSE;
            coef = B(:, idxLambdaMinMSE);
            enet_matrix(:, j) = coef;
            
            lassoPlot(B, FitInfo, 'PlotType', 'Lambda', 'XScale', 'log');
            title(sprintf('%s\nλ=%.4f', output_names{j}, FitInfo.Lambda(idxLambdaMinMSE)), 'FontSize', 9);
            xlabel('正则化参数 λ');
            ylabel('系数');
            
            non_zero_idx = find(abs(coef) > 0.01);
            if ~isempty(non_zero_idx)
                fprintf('\n%s - 弹性网络选择的特征:\n', output_names{j});
                for k = 1:length(non_zero_idx)
                    idx = non_zero_idx(k);
                    fprintf('  %s: %.3f\n', input_names{idx}, coef(idx));
                end
            end
        end
        
        sgtitle('弹性网络回归系数路径');
    else
        fprintf('未找到Statistics and Machine Learning Toolbox，跳过弹性网络分析\n');
    end
catch ME
    fprintf('弹性网络分析失败: %s\n', ME.message);
end

fprintf('\n========== 保存分析结果 ==========\n');

timestamp = datestr(now, 'yyyymmdd_HHMMSS');

results = struct();
results.src_matrix = src_matrix;
results.eta_matrix = eta_matrix;    
results.R2_values = R2_values;
results.p_values = p_values;
results.output_names = output_names;
results.input_names = input_names;
results.mean_sensitivity = mean_sensitivity;
results.amp_sensitivity = amp_sensitivity;
results.diff_sensitivity = diff_sensitivity;

save_filename = sprintf('variance_sensitivity_analysis_%s.mat', timestamp);
save(save_filename, 'results');
fprintf('分析结果已保存到: %s\n', save_filename);

report_filename = sprintf('variance_sensitivity_report_%s.txt', timestamp);
fid = fopen(report_filename, 'w');

fprintf(fid, '输出参数基于方差的灵敏度分析报告\n');
fprintf(fid, '生成时间: %s\n\n', datestr(now));
fprintf(fid, '分析说明:\n');
fprintf(fid, '  本报告分析14个输出参数对18个输入参数的灵敏度\n');
fprintf(fid, '  每个输出参数对应一个点的均值或振幅\n');
fprintf(fid, '  使用标准化回归系数(SRC)和基于方差的敏感性指标(相关比 η²)\n');
fprintf(fid, '  η² 衡量输入对输出的方差解释比例，取值0~1，可捕捉非线性关系\n\n');

fprintf(fid, '各输出参数分析结果:\n');
fprintf(fid, '输出参数 | R²    | 最重要的3个输入参数 (按η²排序)\n');
fprintf(fid, '---------|-------|--------------------------------\n');

for j = 1:num_outputs
    [~, sorted_idx] = sort(eta_matrix(:, j), 'descend');
    top3 = sorted_idx(1:min(3, num_inputs));
    
    fprintf(fid, '%-8s | %.3f | ', output_names{j}, R2_values(j));
    for k = 1:length(top3)
        idx = top3(k);
        star = '';
        if p_values(idx, j) < 0.05
            star = '*';
        end
        fprintf(fid, '%s:η²=%.3f%s ', input_names{idx}, eta_matrix(idx, j), star);
    end
    fprintf(fid, '\n');
end

fprintf(fid, '\n注：*表示线性回归p值<0.05，仅供参考\n');
fprintf(fid, 'η² 越大表示输入对输出的方差影响越大（非线性关系也可捕捉）\n');

fclose(fid);
fprintf('详细报告已保存到: %s\n', report_filename);

fprintf('\n========== 分析完成 ==========\n');

function eta2 = calc_eta_squared(x, y)
   
    n = length(x);

    nbins = max(2, min(10, round(n/50)));  % 对于1000个点，约为10箱
    try
 
        edges = prctile(x, linspace(0,100,nbins+1));
        edges(1) = -inf;
        edges(end) = inf;
        [~, bin] = histc(x, edges);
    catch
    
        [~, bin] = histcounts(x, nbins);
    end
    

    total_ss = sum((y - mean(y)).^2);
    if total_ss == 0
        eta2 = 0;
        return;
    end
    
    within_ss = 0;
    for k = 1:max(bin)
        yk = y(bin == k);
        if length(yk) > 1
            within_ss = within_ss + sum((yk - mean(yk)).^2);
        end
    end
    eta2 = 1 - within_ss / total_ss;

    eta2 = max(0, min(1, eta2));
end