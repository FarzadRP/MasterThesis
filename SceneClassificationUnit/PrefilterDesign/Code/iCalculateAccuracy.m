function [ACC_table] = iCalculateAccuracy(obser_seq, state_seq, estimated_state_seq)
%% Calculating Accuracy (ACC), Detection rate (FAR), False alarm ratev (FAR)

T=size(obser_seq,2);
N=size(obser_seq,1);

%% ACC
acc_amount=0;

for i=1:T
    if state_seq(1,i)==estimated_state_seq(1,i)
        acc_amount=acc_amount+1;
    else
        acc_amount(1,1)=acc_amount+0;
    end
end

acc_overall=acc_amount/T;


%% day = 1
rp_day=0; %right positive
rn_day=0; %right negative
fp_day=0; %false positive
fn_day=0; %false negative

for i=1:T
    if state_seq(1,i)==1 && estimated_state_seq(1,i)==1
        rp_day=rp_day+1;
    elseif state_seq(1,i)~=1 && estimated_state_seq(1,i)~=1
        rn_day=rn_day+1;
    elseif state_seq(1,i)~=1 && estimated_state_seq(1,i)==1
        fp_day=fp_day+1;
    elseif state_seq(1,i)==1 && estimated_state_seq(1,i)~=1
        fn_day=fn_day+1;
    end
end

acc_day=(rp_day+rn_day)/T;
dr_day=(rp_day)/(rp_day+fn_day);
far_day=fp_day/(rn_day+fp_day);

%% night = 2
rp_night=0;
rn_night=0;
fp_night=0;
fn_night=0;

for i=1:T
    if state_seq(1,i)==2 && estimated_state_seq(1,i)==2
        rp_night=rp_night+1;
    elseif state_seq(1,i)~=2 && estimated_state_seq(1,i)~=2
        rn_night=rn_night+1;
    elseif state_seq(1,i)~=2 && estimated_state_seq(1,i)==2
        fp_night=fp_night+1;
    else fn_night=fn_night+1;
    end
end

acc_night=(rp_night+rn_night)/T;
dr_night=(rp_night)/(rp_night+fn_night);
far_night=fp_night/(rn_night+fp_night);

%% rainy = 3
rp_rain=0; %right positive
rn_rain=0; %right negative
fp_rain=0; %false positive
fn_rain=0; %false negative

for i=1:T
    if state_seq(1,i)==3 && estimated_state_seq(1,i)==3
        rp_rain=rp_rain+1;
    elseif state_seq(1,i)~=3 && estimated_state_seq(1,i)~=3
        rn_rain=rn_rain+1;
    elseif state_seq(1,i)~=3 && estimated_state_seq(1,i)==3
        fp_rain=fp_rain+1;
    elseif state_seq(1,i)==3 && estimated_state_seq(1,i)~=3
        fn_rain=fn_rain+1;
    end
end

acc_rain=(rp_rain+rn_rain)/T;
dr_rain=(rp_rain)/(rp_rain+fn_rain);
far_rain=fp_rain/(rn_rain+fp_rain);

ACC_table=[acc_overall, acc_day, dr_day, far_day, acc_night, dr_night, far_night, acc_rain, dr_rain, far_rain];

end

