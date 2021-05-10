function [averagePrecision, recall, precision] = iEvaluateClassifierParallelPerformance()
%% Read dataset and labels
day_testData = load('day_testData.mat'); day_testData = day_testData.day_testData;
night_testData = load('night_testData.mat'); night_testData = night_testData.night_testData;
rainy_testData = load('rainy_testData.mat'); rainy_testData = rainy_testData.rainy_testData;
complete_testData = vertcat(day_testData, night_testData, rainy_testData);

%% Classification----------------------------------------------------------

%% Load saved dataset 
testImageFeaturesDataset = csvread('test_HSV_sharp_hom_corr_contr.csv');

testImageFeaturesDataset(:,5:7) = [];

%% Separate image features from labels
features = 4; labels = 5;
data_test_features = testImageFeaturesDataset(:,1:features); data_test_labels = testImageFeaturesDataset(:,labels);
data_test_features = data_test_features'; data_test_labels = data_test_labels';

%% Prefilter
x = load('x_svm_3', 'x'); x = x.x; x1 = x(50,:);

% Set separators in each image features
[Seg_Hue, Seg_Saturation, Seg_Value, Seg_Sharpness] = iSetSeparatorsInEachObservation(x1);

%% Test data
T = size(data_test_features, 2);

Hue = data_test_features(1,:); 
Saturation = data_test_features(2,:); 
Value = data_test_features(3,:);
Sharpness = data_test_features(4,:);

%% Segmentation of test dataset
hue = iDefineObservationSegments(Hue,Seg_Hue); 
saturation = iDefineObservationSegments(Saturation,Seg_Saturation); 
value = iDefineObservationSegments(Value,Seg_Value);
sharpness = iDefineObservationSegments(Sharpness,Seg_Sharpness);

%% Segmented observations for test
test_segmented_observations =[hue; saturation; value;sharpness] ;
  
%% Classifier
Classifier = load('classifierSVM'); Classifier = Classifier.model;

%% Prediction
predictedLabels = predict(Classifier,test_segmented_observations');

%% Set images into three classes
dayData = []; nightData = []; rainyData = [];
for i = 1:300
    
    if predictedLabels(i) == 1
        dayData = [dayData; complete_testData(i,:)];
    end
    if predictedLabels(i) == 2
        nightData = [nightData; complete_testData(i,:)];
    end
    if predictedLabels(i) == 3
        rainyData = [rainyData; complete_testData(i,:)];
    end
    
end

%% Combine dataset and labels
pp_day_testData = iPrepareDataForTrainTest(dayData);
pp_night_testData = iPrepareDataForTrainTest(nightData);
pp_rainy_testData = iPrepareDataForTrainTest(rainyData);

%% Saved model
% model1 = load('Dmodel'); dayDetector = model1.Dmodel;
% model2 = load('Nmodel'); nightDetector = model2.Nmodel;
% model3 = load('Rmodel'); rainyDetector = model3.Rmodel;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% model1 = load('ODmodel'); dayDetector = model1.ODmodel;
% model2 = load('ONmodel'); nightDetector = model2.ONmodel;
% model3 = load('ORmodel'); rainyDetector = model3.ORmodel;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% model1 = load('DCmodel'); dayDetector = model1.DCmodel;
% model2 = load('NCmodel'); nightDetector = model2.NCmodel;
% model3 = load('RCmodel'); rainyDetector = model3.RCmodel;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model1 = load('ODCmodel'); dayDetector = model1.ODCmodel;
model2 = load('ONCmodel'); nightDetector = model2.ONCmodel;
model3 = load('ORCmodel'); rainyDetector = model3.ORCmodel;

%% Match detection results with ground truth
s = iPrepareMainMatchedDetectionGroundTruth(dayDetector, nightDetector, rainyDetector, pp_day_testData, pp_night_testData, pp_rainy_testData);

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
