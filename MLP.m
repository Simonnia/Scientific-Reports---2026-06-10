clear; clc; close all;
params.numFeatures = 18;    
params.numCurves = 1;      
params.seqLength = 122;     

params.hiddenSize = 128;     
params.numBlocks = 4;      
params.kernelSize = 3;       
params.dilationFactor = 2;   
params.dropoutProb = 0.2;    

params.maxEpochs = 200;     
params.miniBatchSize = 32;   
params.learningRate = 0.001; 
fprintf('输入维度: %d\n', params.numFeatures);
fprintf('输出: %d条曲线, 每条%d个时间点\n', params.numCurves, params.seqLength);
fprintf('TCN块数: %d\n', params.numBlocks);
fprintf('卷积核大小: %d\n', params.kernelSize);
[XTrain, YTrain, XTest, YTest] = generateSingleCurveData(params);
fprintf('训练数据大小: %d 样本\n', size(XTrain, 1));
fprintf('测试数据大小: %d 样本\n', size(XTest, 1));
[XTrainNorm, XTestNorm, YTrainNorm, YTestNorm, inputScaler, outputScaler] = ...
    preprocessSingleCurveData(XTrain, XTest, YTrain, YTest, params);
fprintf('XTrainNorm: %s\n', mat2str(size(XTrainNorm)));
fprintf('YTrainNorm: %s\n', mat2str(size(YTrainNorm)));
lgraph = createSingleCurveTCN(params);
try
    analyzeNetwork(lgraph);
catch ME
    fprintf('网络分析失败: %s\n', ME.message);
    fprintf('继续训练过程...\n');
end
options = trainingOptions('adam', ...
    'MaxEpochs', params.maxEpochs, ...
    'InitialLearnRate', params.learningRate, ...
    'MiniBatchSize', params.miniBatchSize, ...
    'ValidationData', {XTestNorm, YTestNorm}, ...
    'ValidationFrequency', 30, ...
    'Plots', 'training-progress', ...
    'Verbose', true, ...
    'ExecutionEnvironment', 'auto', ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 50, ...
    'GradientThreshold', 1, ...
    'Shuffle', 'every-epoch');
net = trainNetwork(XTrainNorm, YTrainNorm, lgraph, options);
Y_train_pred = predict(net, XTrainNorm);
Y_test_pred = predict(net, XTestNorm);
fprintf('Y_train_pred: %s\n', mat2str(size(Y_train_pred)));
fprintf('Y_test_pred: %s\n', mat2str(size(Y_test_pred)));
Y_train_pred_orig = inverseSingleTransform(Y_train_pred, outputScaler);
Y_train_actual_orig = inverseSingleTransform(YTrainNorm, outputScaler);
Y_test_pred_orig = inverseSingleTransform(Y_test_pred, outputScaler);
Y_test_actual_orig = inverseSingleTransform(YTestNorm, outputScaler);
evaluateSingleCurveModel(Y_train_actual_orig, Y_train_pred_orig, ...
    Y_test_actual_orig, Y_test_pred_orig, params);
visualizeSingleCurveResults(XTrain, YTrain, Y_train_actual_orig, Y_train_pred_orig, ...
    XTest, YTest, Y_test_actual_orig, Y_test_pred_orig, params);
save('single_curve_tcn_model.mat1', 'net', 'params', 'inputScaler', 'outputScaler');
function [XTrain, YTrain, XTest, YTest] = generateSingleCurveData(params)
    numTrainSamples = 800;
    numTestSamples = 200;
           A = xlsread('shuru.xls');
    indices = randperm(1000);
    train_idx = indices(1:800);
    test_idx = indices(801:end);
    
    XTrain = A(train_idx, :);
    XTest = A(test_idx, :);
    

    B = xlsread('0.xlsx');
    YTrain = B(train_idx, 1:122); 
    YTest = B(test_idx, 1:122); 
      end

function [XTrainNorm, XTestNorm, YTrainNorm, YTestNorm, inputScaler, outputScaler] = ...
    preprocessSingleCurveData(XTrain, XTest, YTrain, YTest, params)
      inputScaler.mean = min(min(XTrain), min(XTest));
    inputScaler.std = max(max(XTrain), max(XTest)) - min(min(XTrain), min(XTest));
    XTrainNorm = (XTrain - inputScaler.mean) ./ inputScaler.std;
    XTestNorm = (XTest - inputScaler.mean) ./ inputScaler.std;
      curve_mean = min(min(YTrain), min(YTest));
    curve_std = max(max(YTrain), max(YTest)) - min(min(YTrain), min(YTest));
    outputScaler.curveMeans = curve_mean;
    outputScaler.curveStds = curve_std;
        YTrainNorm = (YTrain - curve_mean) ./ curve_std;
    YTestNorm = (YTest - curve_mean) ./ curve_std;
    XTrainNorm = single(XTrainNorm);
    XTestNorm = single(XTestNorm);
    YTrainNorm = single(YTrainNorm);
    YTestNorm = single(YTestNorm);
    end

function lgraph = createSingleCurveTCN(params)
   
    layers = [
                featureInputLayer(params.numFeatures, 'Name', 'input')
                fullyConnectedLayer(params.hiddenSize * 2, 'Name', 'fc_initial')
        batchNormalizationLayer('Name', 'bn_initial')
        reluLayer('Name', 'relu_initial')
        dropoutLayer(params.dropoutProb, 'Name', 'dropout_initial')
 
        fullyConnectedLayer(params.hiddenSize, 'Name', 'fc1')
        batchNormalizationLayer('Name', 'bn1')
        reluLayer('Name', 'relu1')
        dropoutLayer(params.dropoutProb, 'Name', 'dropout1')
        
             fullyConnectedLayer(256, 'Name', 'fc2')
        batchNormalizationLayer('Name', 'bn2')
        reluLayer('Name', 'relu2')
        dropoutLayer(params.dropoutProb, 'Name', 'dropout2')
        
        fullyConnectedLayer(128, 'Name', 'fc3')
        batchNormalizationLayer('Name', 'bn3')
        reluLayer('Name', 'relu3')
        dropoutLayer(params.dropoutProb, 'Name', 'dropout3')
             fullyConnectedLayer(params.seqLength, 'Name', 'fc_final')
        regressionLayer('Name', 'regression')
    ];
    
    lgraph = layerGraph(layers);
    
end

function output = inverseSingleTransform(input, outputScaler)
       output = input .* outputScaler.curveStds + outputScaler.curveMeans;
    output = double(output);
    
end

function evaluateSingleCurveModel(Y_train_actual, Y_train_pred, Y_test_actual, Y_test_pred, params)
        train_mse = mean((Y_train_actual - Y_train_pred).^2, 'all');
    train_rmse = sqrt(train_mse);
    train_mae = mean(abs(Y_train_actual - Y_train_pred), 'all');
    
    test_mse = mean((Y_test_actual - Y_test_pred).^2, 'all');
    test_rmse = sqrt(test_mse);
    test_mae = mean(abs(Y_test_actual - Y_test_pred), 'all');
    
    fprintf('训练集 - MSE: %.6f, RMSE: %.6f, MAE: %.6f\n', train_mse, train_rmse, train_mae);
    fprintf('测试集 - MSE: %.6f, RMSE: %.6f, MAE: %.6f\n', test_mse, test_rmse, test_mae);
    
    R2 = 1 - sum((Y_test_actual - Y_test_pred).^2) / sum((Y_test_actual - mean(Y_test_actual)).^2);
    fprintf('R²系数: %.4f\n', R2);
   
    results = struct();
    results.overall_metrics = [train_mse, train_rmse, train_mae, test_mse, test_rmse, test_mae];
    save('single_curve_performance.mat', 'results');
end

function visualizeSingleCurveResults(XTrain, YTrain, Y_train_actual, Y_train_pred, ...
    XTest, YTest, Y_test_actual, Y_test_pred, params)
    
      t = 1:params.seqLength;
       figure('Position', [100, 100, 1400, 1000]);
    
       subplot(2, 3, 1);
    scatter(Y_test_actual(:), Y_test_pred(:), 10, 'filled', 'MarkerFaceAlpha', 0.3);
    hold on;
    plot([min(Y_test_actual(:)), max(Y_test_actual(:))], ...
         [min(Y_test_actual(:)), max(Y_test_actual(:))], 'r--', 'LineWidth', 2);
    xlabel('真实值');
    ylabel('预测值');
    title('预测 vs 真实值');
    grid on;
    
    subplot(2, 3, 2);
    residuals = Y_test_actual(:) - Y_test_pred(:);
    histogram(residuals, 50, 'FaceColor', [0.9, 0.5, 0.2], 'FaceAlpha', 0.7);
    xlabel('残差');
    ylabel('频数');
    title('残差分布');
    grid on;
    
    subplot(2, 3, 3);
    num_samples = min(5, size(Y_test_actual, 1));
    sample_indices = randperm(size(Y_test_actual, 1), num_samples);
    
    for i = 1:num_samples
        idx = sample_indices(i);
        plot(t, Y_test_actual(idx, :), '-', 'LineWidth', 1.5, ...
            'DisplayName', sprintf('真实样本%d', idx));
        hold on;
        plot(t, Y_test_pred(idx, :), '--', 'LineWidth', 1, ...
            'DisplayName', sprintf('预测样本%d', idx));
    end
    xlabel('时间点');
    ylabel('数值');
    title('随机样本对比');
    legend('show', 'Location', 'eastoutside');
    grid on;
    
    subplot(2, 3, 4);
    train_errors = mean(abs(Y_train_actual - Y_train_pred), 2);
    test_errors = mean(abs(Y_test_actual - Y_test_pred), 2);
    bar([mean(train_errors), mean(test_errors)]);
    set(gca, 'XTickLabel', {'训练集', '测试集'});
    ylabel('平均绝对误差');
    title('训练集 vs 测试集误差');
    grid on;
    
    subplot(2, 3, 5);
    time_point_errors = mean(abs(Y_test_actual - Y_test_pred), 1);
    plot(t, time_point_errors, 'b-', 'LineWidth', 2);
    xlabel('时间点');
    ylabel('平均绝对误差');
    title('各时间点误差分布');
    grid on;
    
    subplot(2, 3, 6);
    error_matrix = abs(Y_test_actual - Y_test_pred);
    imagesc(error_matrix);
    colorbar;
    xlabel('时间点');
    ylabel('样本编号');
    title('误差热力图');
    
    sgtitle('TCN代理模型综合分析', 'FontSize', 16, 'FontWeight', 'bold');
    
       saveas(gcf, 'single_curve_tcn_analysis.png');
    saveas(gcf, 'single_curve_tcn_analysis.fig');
    

end