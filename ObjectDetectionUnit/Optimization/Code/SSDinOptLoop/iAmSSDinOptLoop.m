function f = iAmSSDinOptLoop(x)
%% Download Pretrained Detector
doTraining = true;
if ~doTraining && ~exist('ssdResNet50VehicleExample_20a.mat','file')
    disp('Downloading pretrained detector (44 MB)...');
    pretrainedURL = 'https://www.mathworks.com/supportfiles/vision/data/ssdResNet50VehicleExample_20a.mat';
    websave('ssdResNet50VehicleExample_20a.mat',pretrainedURL);
end

%% Read dataset and labels
day_trainingData = load('day_trainingData.mat'); day_trainingData = day_trainingData.day_trainingData; 
day_trainingData = day_trainingData(1:166,:);
night_trainingData = load('night_trainingData.mat');  night_trainingData = night_trainingData.night_trainingData;
night_trainingData = night_trainingData(1:167,:);
rainy_trainingData = load('rainy_trainingData.mat'); rainy_trainingData = rainy_trainingData.rainy_trainingData;
rainy_trainingData = rainy_trainingData(1:167,:);
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

pp_day_testData = iPrepareDataForTrainTest(day_testData);
pp_night_testData = iPrepareDataForTrainTest(night_testData);
pp_rainy_testData = iPrepareDataForTrainTest(rainy_testData);
pp_complete_testData = iPrepareDataForTrainTest(complete_testData);

%% Detectors
model = load('smallCompleteDetector.mat'); smallCompleteDetector = model.completeDetector; 

% Focal Loss layer
classes = ["vehicle","Background"]; alpha = x(1,1); gamma = x(1,2);
focalLoss = focalLossLayer('Classes', classes, 'Name', 'focalLoss', 'Alpha', alpha, 'Gamma', gamma);

lgraph = layerGraph(smallCompleteDetector.Network);
lgraph = replaceLayer(lgraph, 'focalLoss', focalLoss);

% Weight freezing
for i = 1:180
    if isprop(lgraph.Layers(i), 'WeightLearnRateFactor')
        
        layerName = lgraph.Layers(i).Name; layerNameChar = char(layerName);
        layerName = lgraph.Layers(i); 
        layerName.WeightLearnRateFactor = 0; layerName.BiasLearnRateFactor = 0;
        layerName.WeightL2Factor = 0; layerName.BiasL2Factor = 0;
        
        lgraph = replaceLayer(lgraph, layerNameChar, layerName);
 
    end
end

%% Train condistions
options = trainingOptions('sgdm', ...
        'MiniBatchSize', 16, ....
        'InitialLearnRate',1e-1, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropPeriod', 30, ...
        'LearnRateDropFactor', 0.8, ...
        'MaxEpochs', 1, ...
        'VerboseFrequency', 50, ...        
        'CheckpointPath', tempdir, ...
        'Shuffle','every-epoch');
    
%% Train
[detector, info] = trainSSDObjectDetector(pp_complete_trainingData, lgraph, options);

%% Test
ap_day = iDetectEvaluate(detector, pp_day_testData);
ap_night = iDetectEvaluate(detector, pp_night_testData);
ap_rainy = iDetectEvaluate(detector, pp_rainy_testData);

%% Objective function
nu = 0;
f(nu+1) = 1 - ap_day;
f(nu+2) = 1 - ap_night;
f(nu+3) = 1 - ap_rainy;



end


