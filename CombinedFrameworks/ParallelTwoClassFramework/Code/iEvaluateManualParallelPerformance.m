function [averagePrecision, recall, precision] = iEvaluateManualParallelPerformance()
%% Read dataset and labels
day_testData = load('day_testData.mat'); day_testData = day_testData.day_testData;
night_testData = load('night_testData.mat'); night_testData = night_testData.night_testData;
rainy_testData = load('rainy_testData.mat'); rainy_testData = rainy_testData.rainy_testData;
complete_testData = vertcat(day_testData, night_testData, rainy_testData);

%% Classification----------------------------------------------------------

%% Or load saved dataset 
testImageFeaturesDataset = csvread('imageFeaturesForTest.csv');

%% Separate image features from labels
features = 1; labels = 16;
data_test_features = testImageFeaturesDataset(:,1:features); data_test_labels = testImageFeaturesDataset(:,labels);
% data_test_features = data_test_features'; data_test_labels = data_test_labels';

%% Classifier
disClassifier = load('twoClassOneImageFeatureDisClassifier', 'disClassifier'); disClassifier = disClassifier.disClassifier;

%% Prediction
predictedLabels = predict(disClassifier,data_test_features);

%% Set images into three classes
dayData = []; nightData = [];
for i = 1:300
    
    if predictedLabels(i) == 1
        dayData = [dayData; complete_testData(i,:)];
    end
    if predictedLabels(i) == 2
        nightData = [nightData; complete_testData(i,:)];
    end
end

%% Combine dataset and labels
pp_day_testData = iPrepareDataForTrainTest(dayData);
pp_night_testData = iPrepareDataForTrainTest(nightData);


%% Saved model
% model1 = load('DRmodel'); model1= model1.DRmodel;
% model2 = load('Nmodel'); model2 = model2.Nmodel;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% model1 = load('ODRmodel'); model1= model1.ODRmodel;
% model2 = load('ONmodel'); model2 = model2.ONmodel;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% model1 = load('DRCmodel'); model1= model1.DRCmodel;
% model2 = load('NCmodel'); model2 = model2.NCmodel;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model1 = load('ODRCmodel'); model1= model1.ODRCmodel;
model2 = load('ONCmodel'); model2 = model2.ONCmodel;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Match detection results with ground truth
s = iPrepareMainMatchedDetectionGroundTruth(model1, model2, pp_day_testData, pp_night_testData);

numClasses       = 1; %numel(classes);
averagePrecision = zeros(numClasses, 1);
precision        = cell(numClasses, 1);
recall           = cell(numClasses, 1);

% Compute the precision and recall for each class
for c = 1 : numClasses
    
    labels = vertcat(s(:,c).labels);
    scores = vertcat(s(:,c).scores);
    numExpected = sum([s(:,c).NumExpected]);
    
    [ap, p, r] = vision.internal.detector.detectorPrecisionRecall(labels, numExpected, scores);
    
    averagePrecision(c) = ap;
    precision{c} = p;
    recall{c}    = r;
end

if numClasses == 1
    precision = precision{1};
    recall    = recall{1};
end

%--------------------------------------------------------------------------
function ds = iDetectionResultsDatastore(results,classname)
if size(results,2) == 2
    % Results has bbox and scores. Add labels. This standardizes the
    % datastore read output to [bbox, scores, labels] and make downstream
    % computation simpler.
    addScores = false;
    addLabels = true;
else
    addScores = false;
    addLabels = false;
end
ds = vision.internal.detector.detectionResultsTableToDatastore(results,addScores,addLabels,classname);
