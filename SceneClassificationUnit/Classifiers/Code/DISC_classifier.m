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

%% discriminant
% rng('default');
DISmodel = fitcdiscr(data_train_,data_train_label,'DiscrimType','pseudoquadratic');

%% Estimation Discriminant
DIS_Yfit0 = predict(DISmodel,data_test_);

%% Evaluation Discriminant
[DIS_ACC_table] = Calculating_Accuracy(data_test_label', data_test_label', DIS_Yfit0');

DIS_rate= DIS_ACC_table(1,1); 
DIS_acc_d= DIS_ACC_table(1,2);
DIS_dr_d= DIS_ACC_table(1,3);
DIS_far_d= DIS_ACC_table(1,4);
DIS_acc_n= DIS_ACC_table(1,5);
DIS_dr_n= DIS_ACC_table(1,6);
DIS_far_n= DIS_ACC_table(1,7);
DIS_acc_r= DIS_ACC_table(1,8);
DIS_dr_r= DIS_ACC_table(1,9);
DIS_far_r= DIS_ACC_table(1,10);

RF_results_day = (1-DIS_acc_d)+(1-DIS_dr_d)+DIS_far_d;
RF_results_night = (1-DIS_acc_n)+(1-DIS_dr_n)+DIS_far_n;
RF_results_rainy = (1-DIS_acc_r)+(1-DIS_dr_r)+DIS_far_r;

%% Save results
save('Results_DISC_IFDS4', 'RF_results_day', 'RF_results_night', 'RF_results_rainy')
