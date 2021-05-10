%% Start 
clc
clear all
close all

%% Download Pretrained Detector
doTraining = true;
if ~doTraining && ~exist('ssdResNet50VehicleExample_20a.mat','file')
    disp('Downloading pretrained detector (44 MB)...');
    pretrainedURL = 'https://www.mathworks.com/supportfiles/vision/data/ssdResNet50VehicleExample_20a.mat';
    websave('ssdResNet50VehicleExample_20a.mat',pretrainedURL);
end

%% Read dataset and labels
day_trainingData = load('day_trainingData.mat'); day_trainingData = day_trainingData.day_trainingData;
night_trainingData = load('night_trainingData.mat');  night_trainingData = night_trainingData.night_trainingData;
rainy_trainingData = load('rainy_trainingData.mat'); rainy_trainingData = rainy_trainingData.rainy_trainingData;
complete_trainingData = vertcat(day_trainingData, night_trainingData, rainy_trainingData);

day_testData = load('day_testData.mat'); day_testData = day_testData.day_testData;
night_testData = load('night_testData.mat'); night_testData = night_testData.night_testData;
rainy_testData = load('rainy_testData.mat'); rainy_testData = rainy_testData.rainy_testData;
complete_testData = vertcat(day_testData, night_testData, rainy_testData);

%% Combine dataset and labels
pp_day_trainingData = iPrepareDataForTrainTest(day_trainingData);
pp_night_trainingData = iPrepareDataForTrainTest(night_trainingData);
pp_rainy_trainingData = iPrepareDataForTrainTest(rainy_trainingData);
pp_complete_trainingData = iPrepareDataForTrainTest(complete_trainingData);

pp_night_testData = iPrepareDataForTrainTest(night_testData);
pp_day_testData = iPrepareDataForTrainTest(day_testData);
pp_rainy_testData = iPrepareDataForTrainTest(rainy_testData);
pp_complete_testData = iPrepareDataForTrainTest(complete_testData);

%% SSD detector
inputSize = [300 300 3]; numClasses = 1; 
lgraph = ssdLayers(inputSize, numClasses, 'resnet50');

%% Training conditions
options = trainingOptions('sgdm', ...
        'MiniBatchSize', 16, ....
        'InitialLearnRate',1e-1, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropPeriod', 30, ...
        'LearnRateDropFactor', 0.8, ...
        'MaxEpochs', 300, ...
        'VerboseFrequency', 50, ...        
        'CheckpointPath', tempdir, ...
        'Shuffle','every-epoch', ...
        'Plots', 'training-progress');
    
%% Train
trainData = pp_day_trainingData; % or pp_night_trainingData or pp_rainy_trainingData or pp_complete_trainingData
[SingleDetector, dayInfo] = trainSSDObjectDetector(trainData, lgraph, options); save('SSD_SingleModel', 'SingleDetector')

%% Test
ap_Detector_dayData = iDetectEvaluate(SingleDetector, pp_day_testData);
ap_Detector_nightData = iDetectEvaluate(SingleDetector, pp_night_testData);
ap_Detector_rainyData = iDetectEvaluate(SingleDetector, pp_rainy_testData);
ap_Detector_completeData = iDetectEvaluate(SingleDetector, pp_complete_testData);

%% Save results
results_SingleDetector = table(ap_Detector_dayData, ap_Detector_nightData, ap_Detector_rainyData, ap_Detector_completeData);
save('results_SingleDetector', 'results_SingleDetector')
