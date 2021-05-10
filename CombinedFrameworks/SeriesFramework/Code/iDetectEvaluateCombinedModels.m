function [averagePrecision, recall, precision] = iDetectEvaluateCombinedModels(model, pp_data)
%% Detection
DetectionResults = detect(model, pp_data, 'Threshold', 0.7); % IOU = 0.4 or 0.5 or 0.7

save('DetectionResults', 'DetectionResults')

%% AP
[averagePrecision, recall, precision] = iEvaluateDetectionPrecisionCombinedModels(DetectionResults, pp_data);

%% logAverageMissRate
% logAverageMissRate = iEvaluateDetectionMissRateCombinedModels(DetectionResults, pp_data);

end