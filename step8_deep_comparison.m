%% STEP 8: COMPREHENSIVE DEEP LEARNING COMPARISON
% Trains/loads CNN, BiLSTM, Autoencoder+DNN and compares with classical ML.

clear; clc;

fprintf('╔══════════════════════════════════════════════════╗\n');
fprintf('║    STEP 8: COMPREHENSIVE DL MODEL COMPARISON    ║\n');
fprintf('╚══════════════════════════════════════════════════╝\n\n');

inputFile = 'features_extracted.mat';
[X, Y] = helper_dl_functions('load_features', inputFile);
[XTrain, YTrain, XVal, YVal, XTest, YTest] = helper_dl_functions('split_data', X, Y, 0.15, 0.20, 42);
[XTrain, XVal, XTest] = helper_dl_functions('standardize', XTrain, XVal, XTest);

if ~isfile('step5_cnn_results.mat')
    run('step5_cnn_model.m');
end
if ~isfile('step6_lstm_results.mat')
    run('step6_lstm_model.m');
end
if ~isfile('step7_autoencoder_results.mat')
    run('step7_autoencoder_model.m');
end

cnnData = load('step5_cnn_results.mat');
lstmData = load('step6_lstm_results.mat');
aeData = load('step7_autoencoder_results.mat');

classicalAcc = helper_dl_functions('train_classical', XTrain, YTrain, XTest, YTest);

modelNames = [{'CNN'; 'BiLSTM'; 'Autoencoder+DNN'}; cellstr(classicalAcc.Model)];
accuracy = [100*cnnData.results.metrics.accuracy; ...
            100*lstmData.results.metrics.accuracy; ...
            100*aeData.results.metrics.accuracy; ...
            classicalAcc.AccuracyPercent];
sensitivity = [cnnData.results.metrics.sensitivity; ...
               lstmData.results.metrics.sensitivity; ...
               aeData.results.metrics.sensitivity; ...
               classicalAcc.Sensitivity];
specificity = [cnnData.results.metrics.specificity; ...
               lstmData.results.metrics.specificity; ...
               aeData.results.metrics.specificity; ...
               classicalAcc.Specificity];
precision = [cnnData.results.metrics.precision; ...
             lstmData.results.metrics.precision; ...
             aeData.results.metrics.precision; ...
             classicalAcc.Precision];
f1score = [cnnData.results.metrics.f1score; ...
           lstmData.results.metrics.f1score; ...
           aeData.results.metrics.f1score; ...
           classicalAcc.F1Score];

comparisonTable = table(modelNames, accuracy, sensitivity, specificity, precision, f1score, ...
    'VariableNames', {'Model','AccuracyPercent','Sensitivity','Specificity','Precision','F1Score'});
comparisonTable = sortrows(comparisonTable, 'AccuracyPercent', 'descend');

figure('Color', 'w', 'Position', [120 120 1200 500]);
subplot(1,2,1);
bar(categorical(comparisonTable.Model), comparisonTable.AccuracyPercent, 'FaceColor', [0.2 0.6 0.8]);
ylabel('Accuracy (%)'); title('Model Accuracy Comparison'); grid on;
xtickangle(25);

subplot(1,2,2);
dlRows = ismember(comparisonTable.Model, {'CNN','BiLSTM','Autoencoder+DNN'});
bar(categorical(comparisonTable.Model(dlRows)), comparisonTable.F1Score(dlRows), 'FaceColor', [0.8 0.4 0.3]);
ylabel('F1-Score'); title('Deep Learning F1 Comparison'); grid on;
xtickangle(25);

saveas(gcf, 'step8_deep_model_comparison.png');
writetable(comparisonTable, 'step8_deep_model_comparison.csv');

results = struct();
results.inputFile = inputFile;
results.comparisonTable = comparisonTable;
results.classicalAccuracy = classicalAcc;
results.cnn = cnnData.results;
results.lstm = lstmData.results;
results.autoencoder = aeData.results;

save('step8_deep_comparison_results.mat', 'results', '-v7.3');

fprintf('✓ Saved comparison MAT: step8_deep_comparison_results.mat\n');
fprintf('✓ Saved comparison CSV: step8_deep_model_comparison.csv\n');
fprintf('✓ Saved comparison plot: step8_deep_model_comparison.png\n\n');

disp('Top models by accuracy:');
disp(comparisonTable(:, {'Model','AccuracyPercent'}));
