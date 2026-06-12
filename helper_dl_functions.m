function varargout = helper_dl_functions(action, varargin)
% HELPER_DL_FUNCTIONS Utility helpers for deep learning EEG pipelines.
% Usage: out = helper_dl_functions('load_features', 'features_extracted.mat');

switch lower(action)
    case 'load_features'
        [varargout{1:nargout}] = dl_load_features(varargin{:});
    case 'split_data'
        [varargout{1:nargout}] = dl_split_data(varargin{:});
    case 'standardize'
        [varargout{1:nargout}] = dl_standardize(varargin{:});
    case 'augment_features'
        [varargout{1:nargout}] = dl_augment_features(varargin{:});
    case 'to_cnn_tensor'
        [varargout{1:nargout}] = dl_to_cnn_tensor(varargin{:});
    case 'to_sequence_cell'
        [varargout{1:nargout}] = dl_to_sequence_cell(varargin{:});
    case 'metrics'
        [varargout{1:nargout}] = dl_metrics(varargin{:});
    case 'plot_history'
        [varargout{1:nargout}] = dl_plot_training_history(varargin{:});
    case 'plot_confusion_roc'
        [varargout{1:nargout}] = dl_plot_confusion_roc(varargin{:});
    case 'train_classical'
        [varargout{1:nargout}] = dl_train_classical_models(varargin{:});
    case 'build_cnn_layers'
        [varargout{1:nargout}] = dl_build_cnn_layers(varargin{:});
    case 'build_bilstm_layers'
        [varargout{1:nargout}] = dl_build_bilstm_layers(varargin{:});
    case 'build_dnn_layers'
        [varargout{1:nargout}] = dl_build_dnn_layers(varargin{:});
    otherwise
        error('Unknown action: %s', action);
end
end

function [X, Y, classNames, dataStruct] = dl_load_features(matFile)
if nargin < 1 || isempty(matFile)
    matFile = 'features_extracted.mat';
end

if ~isfile(matFile)
    error('Input file not found: %s', matFile);
end

dataStruct = load(matFile);
featureCandidates = {'features', 'X', 'feature_table', 'feature_matrix', 'features_extracted', 'all_features'};
labelCandidates = {'labels', 'Y', 'target', 'targets', 'class_labels', 'y'};

X = [];
for i = 1:numel(featureCandidates)
    nm = featureCandidates{i};
    if isfield(dataStruct, nm)
        X = dataStruct.(nm);
        break;
    end
end

if isempty(X)
    vars = fieldnames(dataStruct);
    for i = 1:numel(vars)
        v = dataStruct.(vars{i});
        if isnumeric(v) && ismatrix(v) && min(size(v)) > 1
            X = v;
            break;
        end
    end
end

if istable(X)
    X = table2array(X);
end
X = double(X);

Y = [];
for i = 1:numel(labelCandidates)
    nm = labelCandidates{i};
    if isfield(dataStruct, nm)
        Y = dataStruct.(nm);
        break;
    end
end

if isempty(Y)
    vars = fieldnames(dataStruct);
    for i = 1:numel(vars)
        v = dataStruct.(vars{i});
        if numel(v) == size(X,1)
            Y = v;
            break;
        end
    end
end

if isempty(Y)
    error('No labels found in %s. Expected variables like labels or Y.', matFile);
end

if isrow(Y)
    Y = Y';
end
if isnumeric(Y)
    Y = categorical(Y);
elseif iscell(Y)
    Y = categorical(string(Y));
elseif isstring(Y)
    Y = categorical(Y);
elseif ~iscategorical(Y)
    Y = categorical(Y);
end

if size(X,1) ~= numel(Y)
    if size(X,2) == numel(Y)
        X = X';
    else
        error('Feature-label size mismatch. X rows=%d, labels=%d.', size(X,1), numel(Y));
    end
end

validRows = all(isfinite(X),2) & ~isundefined(Y);
X = X(validRows,:);
Y = removecats(Y(validRows));
classNames = categories(Y);
end

function [XTrain, YTrain, XVal, YVal, XTest, YTest] = dl_split_data(X, Y, valRatio, testRatio, seed)
if nargin < 3 || isempty(valRatio), valRatio = 0.15; end
if nargin < 4 || isempty(testRatio), testRatio = 0.20; end
if nargin < 5 || isempty(seed), seed = 42; end

rng(seed);

try
    cvTest = cvpartition(Y, 'HoldOut', testRatio);
    testMask = test(cvTest);
catch
    idx = randperm(numel(Y));
    nTest = round(testRatio * numel(Y));
    testMask = false(numel(Y),1);
    testMask(idx(1:nTest)) = true;
end

XTest = X(testMask,:);
YTest = Y(testMask);
XRem = X(~testMask,:);
YRem = Y(~testMask);

valFrac = valRatio / (1 - testRatio);
try
    cvVal = cvpartition(YRem, 'HoldOut', valFrac);
    valMask = test(cvVal);
catch
    idx = randperm(numel(YRem));
    nVal = round(valFrac * numel(YRem));
    valMask = false(numel(YRem),1);
    valMask(idx(1:nVal)) = true;
end

XVal = XRem(valMask,:);
YVal = YRem(valMask);
XTrain = XRem(~valMask,:);
YTrain = YRem(~valMask);
end

function [XTrainN, XValN, XTestN, mu, sigma] = dl_standardize(XTrain, XVal, XTest)
mu = mean(XTrain, 1);
sigma = std(XTrain, 0, 1);
sigma(sigma < eps) = 1;

XTrainN = (XTrain - mu) ./ sigma;
XValN = (XVal - mu) ./ sigma;
XTestN = (XTest - mu) ./ sigma;
end

function [XAug, YAug] = dl_augment_features(X, Y, noiseStd, copies)
if nargin < 3 || isempty(noiseStd), noiseStd = 0.02; end
if nargin < 4 || isempty(copies), copies = 1; end
scaleJitterStd = 0.02;

XAug = X;
YAug = Y;
for i = 1:copies
    scale = 1 + scaleJitterStd * randn(size(X,1), 1);
    jitter = noiseStd * randn(size(X));
    XNew = X .* scale + jitter;
    XAug = [XAug; XNew]; %#ok<AGROW>
    YAug = [YAug; Y]; %#ok<AGROW>
end
end

function X4D = dl_to_cnn_tensor(X)
nSamples = size(X,1);
nFeatures = size(X,2);
X4D = reshape(X', [nFeatures, 1, 1, nSamples]);
end

function XSeq = dl_to_sequence_cell(X)
nSamples = size(X,1);
XSeq = cell(nSamples,1);
for i = 1:nSamples
    XSeq{i} = reshape(X(i,:), [1, size(X,2)]);
end
end

function metrics = dl_metrics(YTrue, YPred)
YTrue = categorical(YTrue);
YPred = categorical(YPred);
order = union(categories(YTrue), categories(YPred));
C = confusionmat(YTrue, YPred, 'Order', categorical(order));

N = sum(C(:));
acc = sum(diag(C)) / max(1, N);

numClasses = size(C,1);
sens = zeros(numClasses,1);
spec = zeros(numClasses,1);
prec = zeros(numClasses,1);
f1 = zeros(numClasses,1);

for k = 1:numClasses
    TP = C(k,k);
    FN = sum(C(k,:)) - TP;
    FP = sum(C(:,k)) - TP;
    TN = N - TP - FN - FP;

    sens(k) = TP / max(1, TP + FN);
    spec(k) = TN / max(1, TN + FP);
    prec(k) = TP / max(1, TP + FP);
    f1(k) = 2 * prec(k) * sens(k) / max(eps, prec(k) + sens(k));
end

metrics = struct();
metrics.confusionMatrix = C;
metrics.classOrder = order;
metrics.accuracy = acc;
metrics.sensitivity = mean(sens);
metrics.specificity = mean(spec);
metrics.precision = mean(prec);
metrics.f1score = mean(f1);
end

function dl_plot_training_history(info, figName, saveFile)
if nargin < 2 || isempty(figName), figName = 'Training History'; end
if nargin < 3 || isempty(saveFile), saveFile = 'training_history.png'; end

figure('Name', figName, 'Color', 'w', 'Position', [120 120 1100 450]);
subplot(1,2,1);
plot(info.TrainingLoss, 'LineWidth', 1.8); hold on;
if isfield(info, 'ValidationLoss') && ~isempty(info.ValidationLoss)
    plot(info.ValidationLoss, 'LineWidth', 1.4);
    legend('Training', 'Validation', 'Location', 'best');
else
    legend('Training', 'Location', 'best');
end
title([figName ' - Loss']); xlabel('Iteration'); ylabel('Loss'); grid on;

subplot(1,2,2);
if isfield(info, 'TrainingAccuracy') && ~isempty(info.TrainingAccuracy)
    plot(info.TrainingAccuracy, 'LineWidth', 1.8); hold on;
else
    plot(nan, nan);
end
if isfield(info, 'ValidationAccuracy') && ~isempty(info.ValidationAccuracy)
    plot(info.ValidationAccuracy, 'LineWidth', 1.4);
    legend('Training', 'Validation', 'Location', 'best');
else
    legend('Training', 'Location', 'best');
end
title([figName ' - Accuracy']); xlabel('Iteration'); ylabel('Accuracy (%)'); grid on;

saveas(gcf, saveFile);
end

function dl_plot_confusion_roc(YTrue, YPred, scores, classNames, titleTag, savePrefix)
if nargin < 6, savePrefix = 'dl_model'; end

figure('Name', [titleTag ' Evaluation'], 'Color', 'w', 'Position', [120 120 1100 450]);
subplot(1,2,1);
C = confusionmat(YTrue, YPred, 'Order', categorical(classNames));
imagesc(C); axis tight; colorbar;
set(gca, 'XTick', 1:numel(classNames), 'XTickLabel', classNames, ...
    'YTick', 1:numel(classNames), 'YTickLabel', classNames);
xlabel('Predicted'); ylabel('True'); title([titleTag ' Confusion Matrix']);

subplot(1,2,2);
if numel(classNames) == 2 && ~isempty(scores)
    posClass = categorical(classNames(2));
    [fpr, tpr, ~, auc] = perfcurve(YTrue, scores(:,2), posClass);
    plot(fpr, tpr, 'LineWidth', 2); hold on;
    plot([0 1], [0 1], '--k');
    xlabel('False Positive Rate'); ylabel('True Positive Rate');
    title(sprintf('%s ROC (AUC = %.3f)', titleTag, auc));
    grid on;
else
    text(0.05, 0.5, 'ROC plotted for binary classification only.', 'FontSize', 12);
    axis off;
end

saveas(gcf, [savePrefix '_confusion_roc.png']);
end

function resultTable = dl_train_classical_models(XTrain, YTrain, XTest, YTest)
models = {'SVM', 'RandomForest', 'KNN'};
acc = zeros(numel(models),1);
sens = zeros(numel(models),1);
spec = zeros(numel(models),1);
prec = zeros(numel(models),1);
f1 = zeros(numel(models),1);

svmMdl = fitcecoc(XTrain, YTrain, 'Learners', templateSVM('KernelFunction', 'rbf'));
pred = predict(svmMdl, XTest);
m = dl_metrics(YTest, pred);
acc(1) = 100*m.accuracy;
sens(1) = m.sensitivity;
spec(1) = m.specificity;
prec(1) = m.precision;
f1(1) = m.f1score;

rfMdl = fitcensemble(XTrain, YTrain, 'Method', 'Bag', 'NumLearningCycles', 100);
pred = predict(rfMdl, XTest);
m = dl_metrics(YTest, pred);
acc(2) = 100*m.accuracy;
sens(2) = m.sensitivity;
spec(2) = m.specificity;
prec(2) = m.precision;
f1(2) = m.f1score;

knnMdl = fitcknn(XTrain, YTrain, 'NumNeighbors', 5, 'Distance', 'euclidean');
pred = predict(knnMdl, XTest);
m = dl_metrics(YTest, pred);
acc(3) = 100*m.accuracy;
sens(3) = m.sensitivity;
spec(3) = m.specificity;
prec(3) = m.precision;
f1(3) = m.f1score;

resultTable = table(models', acc, sens, spec, prec, f1, ...
    'VariableNames', {'Model', 'AccuracyPercent', 'Sensitivity', 'Specificity', 'Precision', 'F1Score'});
end

function layers = dl_build_cnn_layers(numFeatures, numClasses)
layers = [
    imageInputLayer([numFeatures 1 1], 'Normalization', 'none', 'Name', 'input')
    convolution2dLayer([5 1], 32, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer([2 1], 'Stride', [2 1], 'Name', 'pool1')
    convolution2dLayer([3 1], 64, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer([2 1], 'Stride', [2 1], 'Name', 'pool2')
    fullyConnectedLayer(96, 'Name', 'fc1')
    dropoutLayer(0.3, 'Name', 'drop1')
    fullyConnectedLayer(numClasses, 'Name', 'fc_out')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classOutput')
    ];
end

function layers = dl_build_bilstm_layers(numClasses)
layers = [
    sequenceInputLayer(1, 'Name', 'input')
    bilstmLayer(96, 'OutputMode', 'sequence', 'Name', 'bilstm1')
    dropoutLayer(0.25, 'Name', 'drop1')
    bilstmLayer(48, 'OutputMode', 'last', 'Name', 'bilstm2')
    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    dropoutLayer(0.3, 'Name', 'drop2')
    fullyConnectedLayer(numClasses, 'Name', 'fc_out')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classOutput')
    ];
end

function layers = dl_build_dnn_layers(inputSize, numClasses)
layers = [
    featureInputLayer(inputSize, 'Normalization', 'none', 'Name', 'input')
    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    dropoutLayer(0.3, 'Name', 'drop1')
    fullyConnectedLayer(32, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(numClasses, 'Name', 'fc_out')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classOutput')
    ];
end
