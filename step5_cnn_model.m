%% STEP 5: CNN MODEL FOR EEG ARTIFACT CLASSIFICATION
% Input: features_extracted.mat (from step 3)
% Output: step5_cnn_results.mat + visualizations

clear; clc;

fprintf('╔══════════════════════════════════════════════════╗\n');
fprintf('║   STEP 5: CNN MODEL FOR EEG CLASSIFICATION      ║\n');
fprintf('╚══════════════════════════════════════════════════╝\n\n');

inputFile = 'features_extracted.mat';
[X, Y, classNames] = helper_dl_functions('load_features', inputFile);

fprintf('✓ Loaded features from %s\n', inputFile);
fprintf('  Samples: %d, Features: %d, Classes: %d\n\n', size(X,1), size(X,2), numel(classNames));

[XTrain, YTrain, XVal, YVal, XTest, YTest] = helper_dl_functions('split_data', X, Y, 0.15, 0.20, 42);
[XTrain, XVal, XTest] = helper_dl_functions('standardize', XTrain, XVal, XTest);
[XTrainAug, YTrainAug] = helper_dl_functions('augment_features', XTrain, YTrain, 0.02, 1);

XTrainCNN = helper_dl_functions('to_cnn_tensor', XTrainAug);
XValCNN = helper_dl_functions('to_cnn_tensor', XVal);
XTestCNN = helper_dl_functions('to_cnn_tensor', XTest);

layers = helper_dl_functions('build_cnn_layers', size(XTrain,2), numel(classNames));

opts = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 35, ...
    'MiniBatchSize', 64, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XValCNN, YVal}, ...
    'ValidationFrequency', 15, ...
    'Verbose', false, ...
    'Plots', 'none');

[net, info] = trainNetwork(XTrainCNN, YTrainAug, layers, opts);

[YPred, scores] = classify(net, XTestCNN);
metrics = helper_dl_functions('metrics', YTest, YPred);

fprintf('CNN Metrics:\n');
fprintf('  Accuracy    : %.2f%%\n', 100*metrics.accuracy);
fprintf('  Sensitivity : %.4f\n', metrics.sensitivity);
fprintf('  Specificity : %.4f\n', metrics.specificity);
fprintf('  Precision   : %.4f\n', metrics.precision);
fprintf('  F1-Score    : %.4f\n\n', metrics.f1score);

helper_dl_functions('plot_history', info, 'CNN Training History', 'step5_cnn_training_history.png');
helper_dl_functions('plot_confusion_roc', YTest, YPred, scores, classNames, 'CNN', 'step5_cnn');

results = struct();
results.modelName = 'CNN';
results.inputFile = inputFile;
results.classNames = classNames;
results.network = net;
results.metrics = metrics;
results.testScores = scores;
results.testPredictions = YPred;
results.testLabels = YTest;
results.trainInfo = info;

save('step5_cnn_results.mat', 'results', '-v7.3');

fprintf('✓ Saved model/results: step5_cnn_results.mat\n');
fprintf('✓ Saved plots: step5_cnn_training_history.png, step5_cnn_confusion_roc.png\n');
fprintf('Expected accuracy target: ~92-96%%\n');
