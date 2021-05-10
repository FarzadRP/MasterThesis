function s = iPrepareMainMatchedDetectionGroundTruth(dayDetector, nightDetector, pp_day_testData, pp_night_testData)

IOU = 0.7;

DetectionResultsDay = detect(dayDetector, pp_day_testData, 'Threshold', IOU);
DetectionResultsNight = detect(nightDetector, pp_night_testData, 'Threshold', IOU);

sDay = iMatchDetectionGroundTruth(DetectionResultsDay, pp_day_testData, IOU);
sNight = iMatchDetectionGroundTruth(DetectionResultsNight, pp_night_testData, IOU);

%% Main matched detection groundTruth
s = vertcat(sDay, sNight);

end
