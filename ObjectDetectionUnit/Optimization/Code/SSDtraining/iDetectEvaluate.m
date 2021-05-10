function ap = iDetectEvaluate(model, pp_data)

DetectionResults = detect(model, pp_data, 'Threshold', 0.4);
ap = evaluateDetectionPrecision(DetectionResults, pp_data);

end