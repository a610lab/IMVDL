function [W,G] = InterationW_BNL(L,X,H,nline,gamma,m,p)
% L: laplacian matrix
% X: data matrix(num*dim)
% H: sample_lack_index
% nline: coefficient of non-linear constrain
% gamma: coefficient of partial L21
% m: projection dimension of W
% p: p value of partial L21

num = size(X{1},1);
dim = size(X{1},2);

Q = eye(dim);
for j=1:length(X)
    Z{j} = nline*X{j}'*diag(H(:,j))*(eye(num)-nline*((2*L+nline*diag(H(:,j))+eps)\diag(H(:,j))))*X{j};
end

INTER_W = 100;
for i = 1:INTER_W
    W_all = [];
    for j=1:length(X)
%         %启用GPU加速
%         ZQ=gpuArray(Z{j}+gamma*Q);
%         [vec,val] = eig(ZQ);
%         vec = gather(vec);
%         val = gather(val);
%         %启用GPU加速
        
        %使用CPU进行计算
        [vec,val] = eig(Z{j}+gamma*Q);
%         [vec,val] = eig((Z{j}+Z{j}')/2+gamma*(Q+Q')/2);
        %使用CPU进行计算
        [~,di] = sort(diag(val));
        W{j} = vec(:,di(1:m));
        W_all = [W_all,W{j}];
        G{j} = nline*((2*L+nline*diag(H(:,j))+eps)\diag(H(:,j)))*X{j}*W{j};
    end
    tempQ = 0.5 * (sqrt(sum(W_all.^2,2)+eps)).^(-1);
    [~,vi] = sort(sqrt(sum(W_all.^2,2)),'descend');
    tempQ(vi(1:p)) = 0;
    Q = diag(tempQ);
    
    w1=0;
    for j=1:length(X)
        w1=w1+2*trace(G{j}'*L*G{j})+nline*norm(diag(H(:,j))*(X{j}*W{j}-G{j}),'fro')^2;
%         w1=w1+2*trace(G{j}'*L*G{j})+nline*norm(diag(H(:,j))*(X{j}*W{j}-G{j}),'fro')^2;
    end
    w2=gamma*sum(sqrt(sum(W_all(vi(p+1:end),:).^2,2)));
%     WResult(i) = sum(sqrt(sum(W_all.^2,2)));
    WResult(i) = w1+w2;

    if i > 1 && abs(WResult(i-1)-WResult(i)) < 0.000001
        break;
    end
end
plot(WResult);
% WResult = WResult';
end