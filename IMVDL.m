function [W,G] = IMVDL(X,sample_lack_index,nline,d,k,gamma,p)
%Input
%X: 1*V cell matrix where each element is num*dim
%sample_lack_index: num*V--1(existing), 0(missing)
%nline: coefficient of non-linear constrain
%d: projection dim of Wm
%k: nearest neighbors
%gamma: coefficient of partial L21
%p: p parameter of partial L21

%Output
%W and G

% 将多模态数据进行正则化 hzw 2023.10.18
num = size(X{1},1);
% for i=1:length(X)
%     X{i}=my_convert2Sparse_impro2(X{i});
% end

% for i=1:length(X)
%     X0 = X{i};
%     mX0 = mean(X0);
%     X1 = X0 - ones(num,1)*mX0;
%     scal = 1./sqrt(sum(X1.*X1)+eps);
%     scalMat = sparse(diag(scal));
%     X{i} = X1*scalMat;
% end

% 初始化相似性矩阵 hzw 202312.2
distX=ComputeSimilarity(X,sample_lack_index);

[distX1, idx] = sort(distX,2);
A = zeros(num);
rr = zeros(num,1);
for i = 1:num
    di = distX1(i,2:k+2);
    rr(i) = 0.5*(k*di(k+1)-sum(di(1:k)));
    id = idx(i,2:k+2);
    A(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
end
r = mean(rr);


A0 = (A+A')/2;
D0 = diag(sum(A0));
L0 = D0 - A0;

% 优化W，partial L21范数+非线性
[W,G] = InterationW_BNL(L0,X,sample_lack_index,nline,gamma,d,p);
NITER = 50;
for iter = 1:NITER
    %更新相似性矩阵
%     distx=zeros(num);
%     for i=1:length(X)
%         distx =distx+L2_distance_1(G{i}',G{i}');
%     end
    distx=ComputeSimilarity(G,sample_lack_index);
    if iter>5
        [~, idx] = sort(distx,2);
    end
    A = zeros(num);
    for i=1:num
        idxa0 = idx(i,2:k+1);
        dxi = distx(i,idxa0);
        ad = -dxi/(2*r);
        A(i,idxa0) = EProjSimplex_new(ad);
    end
    
    A = (A+A')/2;
    D = diag(sum(A));
    L = D-A;
    
    [W,G] = InterationW_BNL(L,X,sample_lack_index,nline,gamma,d,p);
end

% sqW = (W.^2);
% sumW = sum(sqW,2);
% [~,id] = sort(sumW,'descend');


