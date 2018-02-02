%% Upper and lower bounds of tr[E_{\boldsymbol \Phi}(\widetilde{J})^{-1}] versus the compression ratio m/n.
clc;
close all;
clear;

%% Define scenario
n = 80;           % the number of uncompressed measurements
p = 4;             % the number of signals
r = 2;              % the dimension of matrix C is r*p
m_sam = 8:6:n;       % the number of compressed measurements
sigma_e = 1e-1;            % the strength of perturbation
sigma_v = 1e-1;            % the variance of the addictive noise
MC = 5*1e2;            % the number of the Monte Carlo trials
nindex = 0:1:n-1;
nindex = nindex';

A = [exp(1j*2*pi/n*3*nindex),exp(1j*2*pi/n*5*nindex),exp(1j*2*pi/n*7*nindex),exp(1j*2*pi/n*9*nindex)];         % the sensing matrix
C = [zeros(r,p-r),eye(r,r)];             % which can impose some structure on the sensing matrix A
theta = (randn(p,1) + 1i*randn(p,1))/sqrt(2);      % the true value of signals
sigma_w = sigma_e*((C*theta)'*(C*theta)) + sigma_v;  
theta_C = C'*C*theta;
crb = sigma_w*(inv(A'*A)-n*sigma_e^2*((A'*A)\theta_C)*((A'*A)\theta_C)'/...
    (sigma_w+n*sigma_e^2*((theta_C'/(A'*A))*theta_C+(theta_C.'/(A'*A))*conj(theta_C))));         % the CRB before compression

%% Parameter initialization
mmse = zeros(MC,1);
MLE = zeros(length(m_sam),1);
CRB = zeros(length(m_sam),1);
LBmse = zeros(length(m_sam),1);
UPmse = zeros(length(m_sam),1);

%% Calculation
for i=1:length(m_sam)
     m = m_sam(i);
     J_sum = zeros(p,p);
     for mc=1:MC
           v = sqrt(sigma_v/2)*(randn(n,1) + 1i*randn(n,1));       % the addictive noise
           E = sqrt(sigma_e/2)*(randn(n,r) + 1i*randn(n,r));        % the perturbation matrix
           y = (A+E*C)*theta + v;             % the uncompressed measurements
           G=(randn(m,n)+1j*randn(m,n))/sqrt(2);          % the compression matrix
           P=G'/(G*G')*G;
           J_hat_inv = sigma_w*(inv(A'*P*A)-n*sigma_e^2*((A'*P*A)...
               \theta_C)*((A'*P*A)\theta_C)'/(sigma_w+n*sigma_e^2*((theta_C'/(A'*P*A))*theta_C+(theta_C.'/(A'*P*A))*conj(theta_C))));       % the CRB after compression
           J_sum = J_sum + J_hat_inv;       
           
           z = G*y;           % the compressed measurements
           G_hat = (G*G')^(-1/2)*G;
           z_hat = (G*G')^(-1/2)*z;
           A_hat = G_hat*A;
           theta_init = A_hat\z_hat;    % the initial point   
           theta_hat = gradient_descent(theta_init,A_hat,z_hat,sigma_e,C,sigma_v);        % the estimated signals by the gradient descent algorithm
           mmse(mc) = (norm(theta_hat-theta))^2;        % the mean square error 
     end
     J_hat_inv = J_sum/MC;       
     J_hat_inv = (J_hat_inv+conj(J_hat_inv))/2;
     CRB(i) = trace(J_hat_inv);      % the averaged CRB after compression

     LBmatrix = n/m*crb+n^2*(m-n)*sigma_e^2*sigma_w^2*((A'*A)\theta_C)*...
       ((A'*A)\theta_C)'/m/(sigma_w+n*sigma_e^2*((theta_C'/(A'*A))*theta_C+(theta_C.'/(A'*A))*conj(theta_C)))/(m*sigma_w+n^2*sigma_e^2*((theta_C'/(A'*A))*theta_C+(theta_C.'/(A'*A))*conj(theta_C)));
     LBmatrix = (LBmatrix+conj(LBmatrix))/2;
     LBmse(i) = trace(LBmatrix);       % the lower bound
     UPmatrix = (n-p)*crb/(m-p)+n*(n-p)*(m-n)*sigma_w^2*sigma_e^2*((A'*A)\theta_C)*((A'*A)\theta_C)'/(m-p)/(sigma_w+n*sigma_e^2*((theta_C'/(A'*A))*theta_C+(theta_C.'/(A'*A))*conj(theta_C)))/((m-p)*sigma_w+(n-p)*n*sigma_e^2*((theta_C'/(A'*A))*theta_C+(theta_C.'/(A'*A))*conj(theta_C)));
     UPmatrix = (UPmatrix+conj(UPmatrix))/2;
     UPmse(i) = trace(UPmatrix);      % the upper bound
     MLE(i) = mean(mmse);       % the averaged mean square error 
end

%% Plot
alw = 0.75;    % AxesLineWidth
fsz = 10;      % Fontsize
lw = 1.5;      % LineWidth
msz = 8;       % MarkerSize
set(gca, 'FontSize', fsz, 'LineWidth', alw);    %<- Set properties
figure;
plot(m_sam/n,CRB,'-b*',m_sam/n,LBmse,':rs',m_sam/n,UPmse,'-.mo',m_sam/n,MLE,':k+','LineWidth',lw,'MarkerSize',msz);
xlabel('compresion ratio m/n');
ylabel('MSE');
legend('CRB','Lower Bound','Upper Bound','MLE');