function si = iMatchDetectionGroundTruth(detectionResults, groundTruthData, varargin)

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
si = vision.internal.detector.evaluateDetection(resultds, gtds, threshold, classes);

end