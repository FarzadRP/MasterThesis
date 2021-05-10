clc
clear all
close all

data_train = csvread('train_HSV_sharp_hom_corr_contr.csv');
data_test = csvread('test_HSV_sharp_hom_corr_contr.csv');

data_train(:,5:7) = [];
data_test(:,5:7) = [];

a = 4; b = 5;
data_train_ = data_train(:,1:a); data_train_label = data_train(:,b);
data_test_ = data_test(:,1:a); data_test_label = data_test(:,b);

data_train_ = data_train_'; data_train_label = data_train_label';
data_test_ = data_test_'; data_test_label = data_test_label';


%% Prefilter
x1 = load('x_classifier_1'); x1 = x1.x;

for i = 1:height(x1)
    
x1 = load('x_classifier_1'); x1 = x1.x(i,:);

% Theresholds of prefilter
[Seg_Hue, Seg_Saturation, Seg_Value, Seg_Sharpness]  = Seg_6X20(x1);

%% data
T=size(data_train_,2);

Hue = data_train_(1,:);
Saturation = data_train_(2,:);
Value = data_train_(3,:);
Sharpness = data_train_(4,:);

%% Segmentation of training dataset
hue=iDefineObservationSegments(Hue,Seg_Hue);
saturation=iDefineObservationSegments(Saturation,Seg_Saturation);
value=iDefineObservationSegments(Value,Seg_Value);
sharpness=iDefineObservationSegments(Sharpness,Seg_Sharpness);

%% Beobachtung O
train_segmented_observations =[hue; saturation; value; sharpness];

% %% DISC
% rng('default');
% model = fitcdiscr(train_segmented_observations',data_train_label', 'DiscrimType','pseudoquadratic');

% save('classifier_discNew_1', 'model')
%% SVM
rng('default');
SVMtemplate = templateSVM('standardize', true, 'KernelFunction', 'gaussian');
model = fitcecoc(train_segmented_observations', data_train_label', 'Learners', SVMtemplate );

save('classifier2_svm_3', 'model')

% Theresholds of prefilter
[Seg_Hue, Seg_Saturation, Seg_Value, Seg_Sharpness]  = iSetSeparatorsInEachObservation(x1);

%% data
T=size(data_test_,2);

Hue = data_test_(1,:);
Saturation = data_test_(2,:);
Value = data_test_(3,:);
Sharpness = data_test_(4,:);

%% Segmentation of training dataset
hue=iTrainEvaluateClassifierWithPrefilter(Hue,Seg_Hue);
saturation=iTrainEvaluateClassifierWithPrefilter(Saturation,Seg_Saturation);
value=iTrainEvaluateClassifierWithPrefilter(Value,Seg_Value);
sharpness=iTrainEvaluateClassifierWithPrefilter(Sharpness,Seg_Sharpness);

%% Beobachtung O
test_segmented_observations =[hue; saturation; value; sharpness];

%% Estimation disc. classifier
DISC_Yfit0 = predict(model,test_segmented_observations');

%% Evaluation
[ACC_table] = iCalculateAccuracy(data_test_label, data_test_label, DISC_Yfit0');

rate= ACC_table(1,1); 
acc_d= ACC_table(1,2);
dr_d= ACC_table(1,3);
far_d= ACC_table(1,4);
acc_n= ACC_table(1,5);
dr_n= ACC_table(1,6);
far_n= ACC_table(1,7);
acc_r= ACC_table(1,8);
dr_r= ACC_table(1,9);
far_r= ACC_table(1,10);

nu=0;
f(nu+1)=(1-acc_d)+(1-dr_d)+far_d;
f(nu+2)=(1-acc_n)+(1-dr_n)+far_n;
f(nu+3)=(1-acc_r)+(1-dr_r)+far_r;

save('f_classifier_test2_1', 'f')

end
