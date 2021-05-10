function [Obser_seq]  = iDefineObservationSegments(data_1,Seg_X)
%% Preparation
T=size(data_1,2); 

%% Observers and limits a & b
X=zeros(1,T);
for i=1:1:T
    if data_1(1, i) <=Seg_X(1,1)
        X(1,i)=1;
    elseif data_1(1,i)>Seg_X(1,1) && data_1(1,i)<=Seg_X(1,2)
        X(1,i)=2;
    elseif data_1(1,i)>Seg_X(1,2) && data_1(1,i)<=Seg_X(1,3)
        X(1,i)=3;     
%     elseif data_1(1,i)>Seg_X(1,3) && data_1(1,i)<=Seg_X(1,4)
%         X(1,i)=4; 
%     elseif data_1(1,i)>Seg_X(1,4) && data_1(1,i)<=Seg_X(1,5)
%         X(1,i)=5;
    else
        X(1,i)=4;
    end
end

Obser_seq=X;

end