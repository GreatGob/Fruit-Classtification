function [label, hisLabel, hisxcentroid, hisycentroid] = K_mean(K,x,y,Iter)
xcentroid = randi([10,250],[K,1]);
ycentroid = randi([10,200],[K,1]);
hisLabel = [];
hisxcentroid = [];
hisycentroid = [];
d = zeros(length(x),3);
for i=1:Iter
    hisxcentroid = [hisxcentroid, xcentroid];
    hisycentroid = [hisycentroid, ycentroid];
    for j=1:K
        d(:,j) = euclidean(x,y,xcentroid(j),ycentroid(j));
    end
    
    [M,label] = min(d,[],2);
    hisLabel = [hisLabel, label];
    %update
    for j=1:K
        xcentroid(j) = uint8(mean(x(find(label==j))));
        ycentroid(j) = uint8(mean(y(find(label==j))));
    end
end
end

function e = euclidean(x,y,xc,yc)
e = [];
[h,w] = size(x);
for i=1:h
    e = [e; sum((x(i)-xc)^2 + (y(i)-yc)^2)];
end
end