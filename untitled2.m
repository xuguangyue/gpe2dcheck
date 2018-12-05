clear
clc

ff = './';

for j=16
    
    if j==0        
        file = [ff,'real2d-den-init'];
    else
        file =[ff,'real2d-den_',num2str(j)];
    end
    
    [x,y,psi]=loadxy(file);
    surf(x,y,abs(psi).^2)
    shading interp 
    colormap(hot)
    
end