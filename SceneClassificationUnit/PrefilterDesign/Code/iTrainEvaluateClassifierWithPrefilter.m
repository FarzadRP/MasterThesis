function f =iTrainEvaluateClassifierWithPrefilter(x)

data_train = csvread('train_HSV_sharp_hom_corr_contr.csv');

data_train(:,5:7) = [];

a = 4; b = 5;
data_train_ = data_train(:,1:a); data_train_label = data_train(:,b);
data_train_ = data_train_'; data_train_label = data_train_label';

%% Prefilter
x1=x(1,1:48);

% Theresholds of prefilter
[Seg_Hue, Seg_Saturation, Seg_Value, Seg_Sharpness]  = iSetSeparatorsInEachObservation(x1);

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
% %  delta = x(1,31);
% %  gamma = x(1,32);
% model = fitcdiscr(train_segmented_observations',data_train_label', 'DiscrimType','pseudoquadratic');

%% SVM
rng('default');
SVMtemplate = templateSVM('standardize', true, 'KernelFunction', 'gaussian');
model = fitcecoc(train_segmented_observations', data_train_label', 'Learners', SVMtemplate );
  
%% Estimation disc. classifier
DISC_Yfit0 = predict(model,train_segmented_observations');

% %% Estimation
% Yfit0 = predict(RFmodel,data_test_');
%     path0 = str2double(Yfit0);
% Estimated_state=path0';

%% Evaluation
[ACC_table] = iCalculateAccuracy(data_train_label, data_train_label, DISC_Yfit0');

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

end
