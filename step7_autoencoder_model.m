%% STEP 7: AUTOENCODER + DNN MODEL
% Input: features_extracted.mat (from step 3)
% Output: step7_autoencoder_results.mat + visualizations

clear; clc;

fprintf('╔══════════════════════════════════════════════════╗\n');
fprintf('║      STEP 7: AUTOENCODER + DNN MODEL            ║\n');
fprintf('╚══════════════════════════════════════════════════╝\n\n');

inputFile = 'features_extracted.mat';
[X, Y, classNames] = helper_dl_functions('load_features', inputFile);

[XTrain, YTrain, XVal, YVal, XTest, YTest] = helper_dl_functions('split_data', X, Y, 0.15, 0.20, 42);
[XTrain, XVal, XTest] = helper_dl_functions('standardize', XTrain, XVal, XTest);

hiddenSize = max(16, round(size(XTrain,2) * 0.5));

autoenc = trainAutoencoder(XTrain', hiddenSize, ...
    'MaxEpochs', 200, ...
    'L2WeightRegularization', 0.002, ...
    'SparsityRegularization', 2, ...
    'SparsityProportion', 0.05, ...
    'ScaleData', false, ...
    'ShowProgressWindow', false);

XTrainEnc = encode(autoenc, XTrain')';
XValEnc = encode(autoenc, XVal')';
XTestEnc = encode(autoenc, XTest')';

reconTest = predict(autoenc, XTest')';
reconLossPerSample = mean((XTest - reconTest).^2, 2);
meanReconLoss = mean(reconLossPerSample);

layers = helper_dl_functions('build_dnn_layers', size(XTrainEnc,2), numel(classNames));
opts = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 40, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XValEnc, YVal}, ...
    'ValidationFrequency', 15, ...
    'Verbose', false, ...
    'Plots', 'none');

[dnnNet, info] = trainNetwork(XTrainEnc, YTrain, layers, opts);

[YPred, scores] = classify(dnnNet, XTestEnc);
metrics = helper_dl_functions('metrics', YTest, YPred);

fprintf('Autoencoder + DNN Metrics:\n');
fprintf('  Accuracy    : %.2f%%\n', 100*metrics.accuracy);
fprintf('  Sensitivity : %.4f\n', metrics.sensitivity);
fprintf('  Specificity : %.4f\n', metrics.specificity);
fprintf('  Precision   : %.4f\n', metrics.precision);
fprintf('  F1-Score    : %.4f\n', metrics.f1score);
fprintf('  Recon Loss  : %.6f\n\n', meanReconLoss);

helper_dl_functions('plot_history', info, 'Autoencoder+DNN Training', 'step7_autoencoder_dnn_history.png');
helper_dl_functions('plot_confusion_roc', YTest, YPred, scores, classNames, 'Autoencoder+DNN', 'step7_autoencoder');

figure('Color', 'w', 'Position', [120 120 1000 380]);
subplot(1,2,1);
plot(reconLossPerSample, 'LineWidth', 1.4);
xlabel('Test Sample'); ylabel('Reconstruction MSE');
title('Reconstruction Loss Per Sample'); grid on;

subplot(1,2,2);
histogram(reconLossPerSample, 25);
xlabel('Reconstruction MSE'); ylabel('Count');
title(sprintf('Loss Distribution (Mean = %.5f)', meanReconLoss)); grid on;
saveas(gcf, 'step7_autoencoder_reconstruction_loss.png');

results = struct();
results.modelName = 'Autoencoder+DNN';
results.inputFile = inputFile;
results.classNames = classNames;
results.autoencoder = autoenc;
results.classifier = dnnNet;
results.metrics = metrics;
results.testScores = scores;
results.testPredictions = YPred;
results.testLabels = YTest;
results.trainInfo = info;
results.reconstructionLossPerSample = reconLossPerSample;
results.reconstructionLossMean = meanReconLoss;

save('step7_autoencoder_results.mat', 'results', '-v7.3');

fprintf('✓ Saved model/results: step7_autoencoder_results.mat\n');
fprintf('✓ Saved plots: step7_autoencoder_dnn_history.png, step7_autoencoder_confusion_roc.png\n');
fprintf('✓ Saved reconstruction plot: step7_autoencoder_reconstruction_loss.png\n');
fprintf('Expected accuracy target: ~91-94%%\n');
