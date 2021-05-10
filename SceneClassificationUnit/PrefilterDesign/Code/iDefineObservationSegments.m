function [Obser_seq]  = iDefineObservationSegments(data_1,iSetSeparatorsInEachObservation)
%% Preparation
T=size(data_1,2); 

%% Observers and limits a & b

X=zeros(1,T);
for i=1:1:T
    if data_1(1, i) <=iSetSeparatorsInEachObservation(1,1)
        X(1,i)=1;
    elseif data_1(1,i)>iSetSeparatorsInEachObservation(1,1) && data_1(1,i)<=iSetSeparatorsInEachObservation(1,2)
        X(1,i)=2;
    elseif data_1(1,i)>iSetSeparatorsInEachObservation(1,2) && data_1(1,i)<=iSetSeparatorsInEachObservation(1,3)
        X(1,i)=3;     
%     elseif data_1(1,i)>iSetSeparatorsInEachObservation(1,3) && data_1(1,i)<=iSetSeparatorsInEachObservation(1,4)
%         X(1,i)=4; 
%     elseif data_1(1,i)>iSetSeparatorsInEachObservation(1,4) && data_1(1,i)<=iSetSeparatorsInEachObservation(1,5)
%         X(1,i)=5;
%     elseif data_1(1,i)>iSetSeparatorsInEachObservation(1,5) && data_1(1,i)<=iSetSeparatorsInEachObservation(1,6)
%         X(1,i)=6; 
%     elseif data_1(1,i)>iSetSeparatorsInEachObservation(1,6) && data_1(1,i)<=iSetSeparatorsInEachObservation(1,7)
%         X(1,i)=7;
%     elseif data_1(1,i)>iSetSeparatorsInEachObservation(1,7) && data_1(1,i)<=iSetSeparatorsInEachObservation(1,8)
%         X(1,i)=8; 
%     elseif data_1(1,i)>iSetSeparatorsInEachObservation(1,8) && data_1(1,i)<=iSetSeparatorsInEachObservation(1,9)
%         X(1,i)=9;
%     elseif data_1(1,i)>iSetSeparatorsInEachObservation(1,9) && data_1(1,i)<=iSetSeparatorsInEachObservation(1,10)
%         X(1,i)=10; 
%     elseif data_1(1,i)>iSetSeparatorsInEachObservation(1,10) && data_1(1,i)<=iSetSeparatorsInEachObservation(1,11)
%         X(1,i)=11; 
%     elseif data_1(1,i)>iSetSeparatorsInEachObservation(1,11) && data_1(1,i)<=iSetSeparatorsInEachObservation(1,12)
%         X(1,i)=12;
    else
        X(1,i)=4;
    end
end

Obser_seq=X;

end
