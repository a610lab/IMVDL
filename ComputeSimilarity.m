function [distX]=ComputeSimilarity(X,H)
% 计算相似性矩阵 hzw 2023.12.2
num = size(X{1},1);
distX=zeros(num);
div_temp=length(X)*ones(num);
for i=1:length(X)
    distance_temp=L2_distance_1(X{i}',X{i}');
    for j=1:num
        for z=1:num
            if H(j,i)~=0 && H(z,i)~=0
                distX(j,z)=distX(j,z)+distance_temp(j,z);
            else
                if j~=z && H(j,i)==0 && H(z,i)==0
                    distX(j,z)=inf;
                end
                div_temp(j,z)=div_temp(j,z)-1;
                if div_temp(j,z)==0
                    div_temp(j,z)=1;
                    distX(j,z)=inf;
                end
            end
        end
    end
end
distX=distX./div_temp;
end
