function pp_data = iPrepareDataForTrainTest(data_)

imdsTrain = imageDatastore(data_{:,'Var1'});
bldsTrain = boxLabelDatastore(data_(:,'vehicle'));

Data = combine(imdsTrain,bldsTrain); inputSize = [300 300 3];

pp_data = transform(Data,@(data)iPreprocessData(data,inputSize));

end