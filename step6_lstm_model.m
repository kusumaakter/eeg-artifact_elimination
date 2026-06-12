%% STEP 6: BIDIRECTIONAL LSTM MODEL
% Input: features_extracted.mat (from step 3)
% Output: step6_lstm_results.mat + visualizations

clear; clc;

fprintf('╔══════════════════════════════════════════════════╗\n');
fprintf('║     STEP 6: BIDIRECTIONAL LSTM MODEL            ║\n');
fprintf('╚══════════════════════════════════════════════════╝\n\n');

inputFile = 'features_extracted.mat';
[X, Y, classNames] = helper_dl_functions('load_features', inputFile);

[XTrain, YTrain, XVal, YVal, XTest, YTest] = helper_dl_functions('split_data', X, Y, 0.15, 0.20, 42);
[XTrain, XVal, XTest] = helper_dl_functions('standardize', XTrain, XVal, XTest);

XTrainSeq = helper_dl_functions('to_sequence_cell', XTrain);
XValSeq = helper_dl_functions('to_sequence_cell', XVal);
XTestSeq = helper_dl_functions('to_sequence_cell', XTest);

layers = helper_dl_functions('build_bilstm_layers', numel(classNames));

opts = trainingOptions('adam', ...
    'InitialLearnRate', 7e-4, ...
    'MaxEpochs', 45, ...
    'MiniBatchSize', 32, ...
    'GradientThreshold', 1, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XValSeq, YVal}, ...
    'ValidationFrequency', 20, ...
    'Verbose', false, ...
    'Plots', 'none');

[net, info] = trainNetwork(XTrainSeq, YTrain, layers, opts);

[YPred, scores] = classify(net, XTestSeq);
metrics = helper_dl_functions('metrics', YTest, YPred);

fprintf('LSTM Metrics:\n');
fprintf('  Accuracy    : %.2f%%\n', 100*metrics.accuracy);
fprintf('  Sensitivity : %.4f\n', metrics.sensitivity);
fprintf('  Specificity : %.4f\n', metrics.specificity);
fprintf('  Precision   : %.4f\n', metrics.precision);
fprintf('  F1-Score    : %.4f\n\n', metrics.f1score);

helper_dl_functions('plot_history', info, 'BiLSTM Training History', 'step6_lstm_training_history.png');
helper_dl_functions('plot_confusion_roc', YTest, YPred, scores, classNames, 'BiLSTM', 'step6_lstm');

comparison = table([100*metrics.accuracy], 'VariableNames', {'LSTM_AccuracyPercent'});
if isfile('step5_cnn_results.mat')
    tmp = load('step5_cnn_results.mat');
    comparison.CNN_AccuracyPercent = 100 * tmp.results.metrics.accuracy;
end

results = struct();
results.modelName = 'BiLSTM';
results.inputFile = inputFile;
results.classNames = classNames;
results.network = net;
results.metrics = metrics;
results.testScores = scores;
results.testPredictions = YPred;
results.testLabels = YTest;
results.trainInfo = info;
results.performanceComparison = comparison;

save('step6_lstm_results.mat', 'results', '-v7.3');

fprintf('✓ Saved model/results: step6_lstm_results.mat\n');
fprintf('✓ Saved plots: step6_lstm_training_history.png, step6_lstm_confusion_roc.png\n');
fprintf('Expected accuracy target: ~94-97%%\n');
