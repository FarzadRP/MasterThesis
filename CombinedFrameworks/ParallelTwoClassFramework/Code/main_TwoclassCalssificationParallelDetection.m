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

%% Detection and evaluation
[AP, R, P] = iEvaluateManualParallelPerformance();

%% Save results
save('Results_ParallelTwoClass', 'AP')
