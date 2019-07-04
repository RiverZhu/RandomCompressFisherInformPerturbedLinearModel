%% The concentration ellipses for the FIM before and after compression.
clc;
clear;
close all;
lw = 1.5;      % LineWidth
fsz = 12;      % Fontsize

%% Define scenario
n = 20;           % the number of uncompressed measurements
p = 1;             % the number of signals
m = 10;          % the number of compressed measurements
sigma_e = 0.1;             % the strength of perturbation 
sigma_v = 0.1;             % the variance of the addictive noise
nindex = 0:1:n-1;
nindex = nindex';
theta = 0.5+1j;             % the true value of theta
A = exp(1j*2*pi/n*3*nindex);              % the true value of the sensing matrix
C = 1;                          % which can impose some structure on the sensing matrix A
theta_C = C'*C*theta;  
theta_C_underline =[theta_C;conj(theta_C)];
sigma_w = sigma_e*((C*theta)'*(C*theta))+sigma_v;

%% Plot the concentration ellipse for the FIM before compression
J = [A'*A/sigma_w+sigma_e^2*n*(theta_C*theta_C')/(sigma_w^2),sigma_e^2*n*(theta_C*theta_C.')/(sigma_w^2);...
    sigma_e^2*n*(conj(theta_C)*theta_C')/(sigma_w^2),conj(A'*A/sigma_w)+sigma_e^2*n*(conj(theta_C)*theta_C.')/(sigma_w^2)];      
% the FIM before compression
f1=@(x1,y1)(J(1,1)+J(2,2)+2*real(J(1,2)))*(x1.^2)+(J(1,1)+J(2,2)-2*real(J(1,2)))*(y1.^2)+4*x1.*y1*imag(J(1,2))-J(1,1);
h1=fimplicit(f1,[-2,2,-2,2]);
set(h1,'Color','r','LineWidth',lw,'linestyle','-');
hold on;

%% Plot the concentration ellipse for the FIM after compression
MC = 50;                     % the number of realizations of the FIM after compression
for k = 1:MC
     G=(randn(m,n)+1j*randn(m,n))/sqrt(2);          % the compression matrix
     P=G'/(G*G')*G;
     M_P = A'*P*A;
     J_hat = [M_P/sigma_w+sigma_e^2*n*(theta_C*theta_C')/(sigma_w^2),sigma_e^2*n*(theta_C*theta_C.')/(sigma_w^2);...
        sigma_e^2*n*(conj(theta_C)*theta_C')/(sigma_w^2),conj(M_P/sigma_w)+sigma_e^2*n*(conj(theta_C)*theta_C.')/(sigma_w^2)];     
    % the FIM after compression
     J_hat = (J_hat + J_hat')/2;   % symmetry operation
     f2=@(x2,y2)(J_hat(1,1)+J_hat(2,2)+2*real(J_hat(1,2)))*(x2.^2)+(J_hat(1,1)+J_hat(2,2)-2*real(J_hat(1,2)))*(y2.^2)+4*x2.*y2*imag(J_hat(1,2))-J(1,1);
     h2=fimplicit(f2,[-2,2,-2,2]);
     set(h2,'Color','b','LineWidth',lw,'linestyle',':');
     waitbar(k/MC)
end

delete(get(gca,'title'));
delete(get(gca,'xlabel'));
delete(get(gca,'ylabel'));
xlabel('e_r','Fontsize',fsz);
ylabel('e_i','Fontsize',fsz);
legend('Before compression','After compression');
set(gca, 'FontSize', fsz,'FontName','Times New Roman');
save figure1.mat