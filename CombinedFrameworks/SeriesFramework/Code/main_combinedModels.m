%% start 
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
day_testData = load('day_testData.mat'); day_testData = day_testData.day_testData;
night_testData = load('night_testData.mat'); night_testData = night_testData.night_testData;
rainy_testData = load('rainy_testData.mat'); rainy_testData = rainy_testData.rainy_testData;
complete_testData = vertcat(day_testData, night_testData, rainy_testData);

%% Combine dataset and labels
pp_night_testData = iPrepareDataForTrainTest(night_testData);
pp_day_testData = iPrepareDataForTrainTest(day_testData);
pp_rainy_testData = iPrepareDataForTrainTest(rainy_testData);
pp_complete_testData = iPrepareDataForTrainTest(complete_testData);

%% Saved model
% model5 = load('Cmodel'); detector = model5.Cmodel;
model5 = load('OCmodel'); detector = model5.OCmodel;

%% Evaluation
[AP, R, P] = iDetectEvaluateCombinedModels(detector, pp_complete_testData);

%% Save results
save('Results_seriesOptimized', 'AP');

