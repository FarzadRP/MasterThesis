function [Seg_1, Seg_2, Seg_3,Seg_4] = iSetSeparatorsInEachObservation(x)

fi = 3;
Seg_1=sort(x(1,1:fi));
Seg_2=sort(x(1,fi+1:2*fi));
Seg_3=sort(x(1,(2*fi)+1:3*fi));
Seg_4=sort(x(1,(3*fi)+1:4*fi));
% Seg_5=sort(x(1,(4*fi)+1:5*fi));
% Seg_6=sort(x(1,(5*fi)+1:6*fi));
% Seg_7=sort(x(1,(6*fi)+1:7*fi));
% Seg_8=sort(x(1,(7*fi)+1:8*fi));
% Seg_9=sort(x(1,(8*fi)+1:9*fi));
% Seg_10=sort(x(1,(9*fi)+1:10*fi));
% Seg_11=sort(x(1,(10*fi)+1:11*fi));
% Seg_12=sort(x(1,(11*fi)+1:12*fi));
% Seg_13=sort(x(1,(12*fi)+1:13*fi));

end