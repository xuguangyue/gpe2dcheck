clear
clc

ff = './';

for j=44
    
    if j==0        
        file = [ff,'real2d-den-init'];
    else
        file =[ff,'real2d-den_',num2str(j)];
    end
    
    [x,y,psi]=loadxy(file);
    surf(x,y,log10(abs(psi).^2))
    shading interp 
    colormap(hot)
    view(0,90)
    axis equal
    xlim([-32 32])
    ylim([-32 32])
    
end