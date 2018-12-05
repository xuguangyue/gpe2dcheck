clear
clc

dyn = load('real2d-dyn.txt');

t   = dyn(:,1);
nt  = length(t);
dt  = t(2) - t(1);
dx2 = dyn(:,4);
dy2 = dyn(:,6);

% t   = dyn(1:1000,1);
% nt  = length(t);
% dt  = t(2) - t(1);
% dx2 = dyn(1:1000,4);
% dy2 = dyn(1:1000,6);


% t   = dyn(1001:2000,1);
% nt  = length(t);
% dt  = t(2) - t(1);
% dx2 = dyn(1001:2000,4);
% dy2 = dyn(1001:2000,6);

dxw = fftshift(fft(dx2 - mean(dx2)));
dyw = fftshift(fft(dy2 - mean(dy2)));

if ~mod(nt,2)
    w = 2*pi/(nt*dt)*(-nt/2:nt/2-1);
else
    w = 2*pi/(nt*dt)*(-(nt-1)/2:(nt-1)/2);
end

subplot(211)
plot(t,dx2,t,dy2)
xlabel('\omega_0 t')
ylabel('\Deltax/l_0')
grid on
subplot(212)
plot(w,abs(dxw),w,abs(dyw))
xlabel('$\omega/\omega_0$','interpreter','latex')
ylabel('\Deltax(\omega)')
xlim([-5 5])
grid on