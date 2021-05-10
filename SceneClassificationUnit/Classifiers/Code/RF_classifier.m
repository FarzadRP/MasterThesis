clc
close all
clear all

data_train = csvread('train_HSV_sharp_hom_corr_contr.csv');
data_test = csvread('test_HSV_sharp_hom_corr_contr.csv');

% data_train(:,7) = [];
% data_test(:,7) = [];

a = 7; b = 8;
data_train_ = data_train(:,1:a); data_train_label = data_train(:,b);
data_test_ = data_test(:,1:a); data_test_label = data_test(:,b);

%% RF
rng('default');
RFmodel = TreeBagger(46,data_train_,data_train_label,'Method','classification');

%% Estimation RF
RF_Yfit0 = predict(RFmodel,data_test_);
    RF_path0 = str2double(RF_Yfit0);
RF_Estimated_state=RF_path0;

%% Evaluation RF
[RF_ACC_table] = Calculating_Accuracy(data_test_label', data_test_label', RF_Estimated_state');

RF_rate= RF_ACC_table(1,1); 
RF_acc_d= RF_ACC_table(1,2);
RF_dr_d= RF_ACC_table(1,3);
RF_far_d= RF_ACC_table(1,4);
RF_acc_n= RF_ACC_table(1,5);
RF_dr_n= RF_ACC_table(1,6);
RF_far_n= RF_ACC_table(1,7);
RF_acc_r= RF_ACC_table(1,8);
RF_dr_r= RF_ACC_table(1,9);
RF_far_r= RF_ACC_table(1,10);

RF_ob_fun_day = (1-RF_acc_d)+(1-RF_dr_d)+RF_far_d;
RF_ob_fun_night = (1-RF_acc_n)+(1-RF_dr_n)+RF_far_n;
RF_ob_fun_rainy = (1-RF_acc_r)+(1-RF_dr_r)+RF_far_r;

%% Save results
save('Results_RF_IFDS4', 'RF_ob_fun_day', 'RF_ob_fun_night', 'RF_ob_fun_rainy')
