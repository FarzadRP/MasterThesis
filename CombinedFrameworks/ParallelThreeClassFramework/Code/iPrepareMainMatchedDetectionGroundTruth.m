function s = iPrepareMainMatchedDetectionGroundTruth(dayDetector, nightDetector, rainyDetector, pp_day_testData, pp_night_testData, pp_rainy_testData)

t = 0.7;

DetectionResultsDay = detect(dayDetector, pp_day_testData, 'Threshold', t);
DetectionResultsNight = detect(nightDetector, pp_night_testData, 'Threshold', t);
DetectionResultsRainy = detect(rainyDetector, pp_rainy_testData, 'Threshold', t);

sDay = iMatchDetectionGroundTruth(DetectionResultsDay, pp_day_testData, t);
sNight = iMatchDetectionGroundTruth(DetectionResultsNight, pp_night_testData, t);
sRainy = iMatchDetectionGroundTruth(DetectionResultsRainy, pp_rainy_testData, t);

%% Main matched detection groundTruth
s = vertcat(sDay, sNight, sRainy);

end
