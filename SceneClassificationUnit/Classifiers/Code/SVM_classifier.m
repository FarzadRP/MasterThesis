clc
close all
clear all

data_train = csvread('train_HSV_sharp_hom_corr_contr.csv');
data_test = csvread('test_HSV_sharp_hom_corr_contr.csv');

% data_train(:,7) = [];
% data_test(:,7) = [];

a =7; b = 8;
data_train_ = data_train(:,1:a); data_train_label = data_train(:,b);
data_test_ = data_test(:,1:a); data_test_label = data_test(:,b);

%% SVM
rng('default');
SVMtemplate = templateSVM('standardize', true, 'KernelFunction', 'gaussian');
SVMmodel = fitcecoc(data_train_, data_train_label, 'Learners', SVMtemplate );

%% Estimation SVM
SVM_Yfit0 = predict(SVMmodel,data_test_);

%% Evaluation SVM
[SVM_ACC_table] = Calculating_Accuracy(data_test_label', data_test_label', SVM_Yfit0');

SVM_rate= SVM_ACC_table(1,1); 
SVM_acc_d= SVM_ACC_table(1,2);
SVM_dr_d= SVM_ACC_table(1,3);
SVM_far_d= SVM_ACC_table(1,4);
SVM_acc_n= SVM_ACC_table(1,5);
SVM_dr_n= SVM_ACC_table(1,6);
SVM_far_n= SVM_ACC_table(1,7);
SVM_acc_r= SVM_ACC_table(1,8);
SVM_dr_r= SVM_ACC_table(1,9);
SVM_far_r= SVM_ACC_table(1,10);

SVM_ob_fun_day = (1-SVM_acc_d)+(1-SVM_dr_d)+SVM_far_d;
SVM_ob_fun_night = (1-SVM_acc_n)+(1-SVM_dr_n)+SVM_far_n;
SVM_ob_fun_rainy = (1-SVM_acc_r)+(1-SVM_dr_r)+SVM_far_r;

%% Save results
save('Results_SVM_IFDS4', 'SVM_ob_fun_day', 'SVM_ob_fun_night', 'SVM_ob_fun_rainy')
