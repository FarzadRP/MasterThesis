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

%% Knn
rng('default');
KNNmodel = fitcknn(data_train_, data_train_label, 'NumNeighbors',7,'Standardize',1);

%% Estimation KNN
KNN_Yfit0 = predict(KNNmodel,data_test_);

%% Evaluation KNN
[KNN_ACC_table] = Calculating_Accuracy(data_test_label', data_test_label', KNN_Yfit0');

KNN_rate= KNN_ACC_table(1,1); 
KNN_acc_d= KNN_ACC_table(1,2);
KNN_dr_d= KNN_ACC_table(1,3);
KNN_far_d= KNN_ACC_table(1,4);
KNN_acc_n= KNN_ACC_table(1,5);
KNN_dr_n= KNN_ACC_table(1,6);
KNN_far_n= KNN_ACC_table(1,7);
KNN_acc_r= KNN_ACC_table(1,8);
KNN_dr_r= KNN_ACC_table(1,9);
KNN_far_r= KNN_ACC_table(1,10);

KNN_ob_fun_day = (1-KNN_acc_d)+(1-KNN_dr_d)+KNN_far_d;
KNN_ob_fun_night = (1-KNN_acc_n)+(1-KNN_dr_n)+KNN_far_n;
KNN_ob_fun_rainy = (1-KNN_acc_r)+(1-KNN_dr_r)+KNN_far_r;

%% Save results
save('Results_KNN_IFDS4', 'KNN_ob_fun_day', 'KNN_ob_fun_night', 'KNN_ob_fun_rainy')

