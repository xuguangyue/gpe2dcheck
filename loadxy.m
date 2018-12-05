function [x,y,var]=loadxy(name)

data = dlmread([name,'.txt']);
x  = unique(data(:,1));
Nx = length(x);
y  = unique(data(:,2));
Ny = length(y);
[~,ss]=size(data);
if ss==4
    var=reshape(data(:,ss-1)+1i*data(:,ss),Ny,Nx);
else
    if ss==3 
        var=reshape(data(:,ss),Ny,Nx);
    end
end

end