function [averagePrecision, recall, precision] = iEvaluateDetectionPrecisionCombinedModels(...
    detectionResults, groundTruthData, varargin)
%evaluateDetectionPrecision Evaluate the precision metric for object detection.
%   averagePrecision = evaluateDetectionPrecision(detectionResults,
%   groundTruthData) returns average precision to measure the detection
%   performance. For a multi-class detector, averagePrecision is a vector
%   of average precision scores for each object class. The class order
%   follows the same column order as the groundTruthData table.
%
%   This function supports evaluation of object detectors that output
%   bounding box locations as axis-aligned rectangles. To evaluate
%   detectors that output bounding box locations as rotated rectangles, use
%   evaluateDetectionAOS.
%
%   Inputs:
%   -------
%   detectionResults  - a table that has two columns for single-class
%                       detector, or three columns for multi-class
%                       detector. The first column contains M-by-4 matrices
%                       of [x, y, width, height] bounding boxes specifying
%                       object locations. The second column contains scores
%                       for each detection. For multi-class detector, the
%                       third column contains the predicted label for each
%                       detection. The label must be categorical type
%                       defined by the variable names of groundTruthData
%                       table.
%
%   groundTruthData   - This must be a datastore or a table.
%
%                       Datastore format:
%                       -----------------
%                       A datastore that returns a table or cell array on the read
%                       methods with two columns or three columns. In case of three
%                       columns, the first column is ignored.
%                       1st Column: A cell vector that contain M-by-4 matrices of [x,
%                                   y, width, height] bounding boxes specifying object
%                                   locations within each image. Each 1-by-4 represents
%                                   a single object class, e.g. person, car, dog.
%                       2nd Column: A categorical vector of size M-by-1 containing the
%                                   object class names. Note that all the categorical
%                                   data returned by the datastore must have the same
%                                   categories.
%                       Use boxLabelDatastore that can return the 1st and 2nd column of data.
%
%                       Table format:
%                       -------------
%                       A table that has one column for single-class, or
%                       multiple columns for multi-class. Each column
%                       contains M-by-4 matrices of [x, y, width, height]
%                       bounding boxes specifying object locations. The
%                       column name specifies the class label.
%
%   [..., recall, precision] = evaluateDetectionPrecision(...) returns data
%   points for plotting the precision/recall curve. You can visualize the
%   performance curve using plot(recall, precision). For multi-class
%   detector, recall and precision are cell arrays, where each cell
%   contains the data points for each object class.
%
%   [...] = evaluateDetectionPrecision(..., threshold) specifies the
%   overlap threshold for assigning a detection to a ground truth box. The
%   overlap ratio is computed as the intersection over union. The default
%   value is 0.5.
%
%   Example: Evaluate an YOLOv2 vehicle detector
%   ---------------------------------------------
%
%   % Load a table containing the training data.
%   % The first column contains the training images, the remaining columns
%   % contain the labeled bounding boxes.
%   data = load('vehicleTrainingData.mat');
%   trainingData = data.vehicleTrainingData(1:10,:);
%
%   % Load the detector containing the layerGraph for trainining.
%   vehicleDetector = load('yolov2VehicleDetector.mat');
%
%   % Add fullpath to the local vehicle data folder.
%   dataDir = fullfile(toolboxdir('vision'), 'visiondata');
%   trainingData.imageFilename = fullfile(dataDir, trainingData.imageFilename);
%
%   % Create an imageDatastore using the files from the table.
%   imds = imageDatastore(trainingData.imageFilename);
%   % Create a boxLabelDatastore using the label columns from the table.
%   blds = boxLabelDatastore(trainingData(:,2:end));
%
%   % Run the detector with imageDatastore.
%   detector = vehicleDetector.detector;
%   results = detect(detector, imds);
%
%   % Evaluate the results against the ground truth data
%   [ap, recall, precision] = evaluateDetectionPrecision(results, blds);
%
%   % Plot precision/recall curve
%   figure
%   plot(recall, precision)
%   grid on
%   title(sprintf('Average precision = %.1f', ap))
%
% See also evaluateDetectionMissRate, evaluateDetectionAOS,
%          bboxOverlapRatio, boxLabelDatastore.

% Copyright 2016-2019 The MathWorks, Inc.
%
% References
% ----------
%   [1] C. D. Manning, P. Raghavan, and H. Schutze. An Introduction to
%   Information Retrieval. Cambridge University Press, 2008.
%
%   [2] D. Hoiem, Y. Chodpathumwan, and Q. Dai. Diagnosing error in
%   object detectors. In Proc. ECCV, 2012.
%
%   [3] Dollar, Piotr, et al. "Pedestrian Detection: An Evaluation of the
%   State of the Art." Pattern Analysis and Machine Intelligence, IEEE
%   Transactions on 34.4 (2012): 743 - 761.

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
s = vision.internal.detector.evaluateDetection(resultds, gtds, threshold, classes); 

day_testData = load('day_testData.mat'); day_testData = day_testData.day_testData;
night_testData = load('night_testData.mat'); night_testData = night_testData.night_testData;
rainy_testData = load('rainy_testData.mat'); rainy_testData = rainy_testData.rainy_testData;
complete_testData = vertcat(day_testData, night_testData, rainy_testData); 

% model3 = load('Dmodel'); detector = model3.Dmodel;
model3 = load('ODmodel'); detector = model3.ODmodel;

for i = 1:300
    
    values = s(i,1).labels;
    values_sum = sum(values);
    
    if values_sum == 0
        
        complete_testData_forModel_3 = complete_testData(i,:);
        pp_complete_testData_forModel3 = iPrepareDataForTrainTest(complete_testData_forModel_3);
        
        DetectionResults_Model3 = detect(detector, pp_complete_testData_forModel3, 'Threshold', 0.7);
        s3 = iEvaluateDetectionPrecisionSupporterModel(DetectionResults_Model3, pp_complete_testData_forModel3);

        s(i,1).labels = s3.labels;
        s(i,1).scores = s3.scores;
        s(i,1).Detections = s3.Detections;
        s(i,1).FalseNegative = s3.FalseNegative;
        s(i,1).GroundTruthAssignments = s3.GroundTruthAssignments;
        s(i,1).NumExpected = s3.NumExpected;
        
    end
end

numClasses       = numel(classes);
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
