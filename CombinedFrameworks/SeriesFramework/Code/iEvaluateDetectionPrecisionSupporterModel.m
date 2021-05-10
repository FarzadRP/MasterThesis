function s3 = iEvaluateDetectionPrecisionSupporterModel(...
    detectionResults, groundTruthData, varargin)

narginchk(2, 3);

% Validate user inputs
boxFormat = 4; % axis-aligned
[gtds, classes] = vision.internal.detector.evaluationInputValidation(detectionResults, ...
    groundTruthData, mfilename, true, boxFormat, varargin{:});

% Hit/miss threshold for IOU (intersection over union) metric.
threshold = 0.5;
if ~isempty(varargin)
    threshold = varargin{1};
end

resultds = iDetectionResultsDatastore(detectionResults,classes);

% Match the detection results with ground truth
s3 = vision.internal.detector.evaluateDetection(resultds, gtds, threshold, classes);


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

end
