%% The asymptotic relative efficiency versus the strength of perturbation \sigma_e^2.
clc;
close all;
clear;

%% Define scenario
n = 64;           % the number of uncompressed measurements
p = 8;             % the number of signals
r = 2;              % the dimension of matrix C is r*p
m = n/2;         % the number of compressed measurements
sigma_v = 1e-3;           % the variance of the addictive noise
sigma_e_sam = [1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,1e-1,5e-1,1e0,5e0,1e1,5e1,1e2];            % the strength of perturbation
MC = 5*1e3;          % the number of the Monte Carlo trials
nindex = 0:1:n-1;
nindex = nindex';
rindex = linspace(1,2*p-1,p);
A = exp(1j*2*pi/n*nindex*rindex);        % the sensing matrix
C = [zeros(r,p-r),eye(r,r)];         % which can impose some structure on the sensing matrix A
theta = ones(p,1)+1j*0.5*ones(p,1);      % the true value of signals
theta_C = C'*C*theta;

%% Parameter initialization
mmse_compressed = zeros(MC,1);
mmse_uncompressed = zeros(MC,1);
mse_compressed = zeros(length(sigma_e_sam),1);
mse_uncompressed = zeros(length(sigma_e_sam),1);
CRB = zeros(length(sigma_e_sam),1);
LBmse = zeros(length(sigma_e_sam),1);
UPmse = zeros(length(sigma_e_sam),1);
J = zeros(length(sigma_e_sam),1);
J_lower = zeros(length(sigma_e_sam),1);
J_upper = zeros(length(sigma_e_sam),1);
J_mse = zeros(length(sigma_e_sam),1);
crb_trace = zeros(length(sigma_e_sam),1);

%% Calculation
for  i=1:length(sigma_e_sam)
      sigma_e= sigma_e_sam(i);
      sigma_w = sigma_e*((C*theta)'*(C*theta)) + sigma_v;  
      J_sum = zeros(p,p);
      for mc=1:MC
            v = sqrt(sigma_v/2)*(randn(n,1) + 1i*randn(n,1));      % the addictive noise
            E = sqrt(sigma_e/2)*(randn(n,r) + 1i*randn(n,r));       % the perturbation matrix
            y = (A+E*C)*theta + v;            % the uncompressed measurements
            theta_init_un = A\y;      % the initial point
            theta_hat_uncompressed = gradient_descent(theta_init_un,A,y,sigma_e,C,sigma_v);        % the estimated signals by the gradient descent algorithm of the uncompressed model
            mmse_uncompressed(mc) = (norm(theta_hat_uncompressed-theta))^2;       % the mean square error before compression

            G=(randn(m,n)+1j*randn(m,n))/sqrt(2);        % the compression matrix
            P=G'/(G*G')*G;
            J_sum = J_sum + sigma_w*(inv(A'*P*A)-n*sigma_e^2*((A'*P*A)...
               \theta_C)*((A'*P*A)\theta_C)'/(sigma_w+n*sigma_e^2*((theta_C'/(A'*P*A))*theta_C+(theta_C.'/(A'*P*A))*conj(theta_C))));   % the CRB after compression
            
            z = G*y;         % the compressed measurements
            G_hat = (G*G')^(-1/2)*G;
            z_hat = (G*G')^(-1/2)*z;
            A_hat = G_hat*A;            
            theta_init = A_hat\z_hat;          % the initial point                                
            theta_hat_uncompressed = gradient_descent(theta_init,A_hat,z_hat,sigma_e,C,sigma_v);      % the estimated signals by the gradient descent algorithm of the compressed model
            mmse_compressed(mc) = (norm(theta_hat_uncompressed-theta))^2;        % the mean square error after compression
      end
      crb = sigma_w*(inv(A'*A)-n*sigma_e^2*((A'*A)\theta_C)*((A'*A)\theta_C)'/...
               (sigma_w+n*sigma_e^2*((theta_C'/(A'*A))*theta_C+(theta_C.'/(A'*A))*conj(theta_C))));
      crb = (crb+conj(crb))/2;
      crb_trace(i) = trace(crb);      % the CRB before compression
      J_hat_inv = (J_sum+conj(J_sum))/2/MC;
      CRB(i) = trace(J_hat_inv);         % the averaged CRB after compression
   
      UPmatrix = (n-p)*crb/(m-p)+n*(n-p)*(m-n)*sigma_w^2*sigma_e^2*((A'*A)\theta_C)*((A'*A)\theta_C)'/(m-p)/(sigma_w+n*sigma_e^2*((theta_C'/(A'*A))*theta_C+(theta_C.'/(A'*A))*conj(theta_C)))/((m-p)*sigma_w+(n-p)*n*sigma_e^2*((theta_C'/(A'*A))*theta_C+(theta_C.'/(A'*A))*conj(theta_C)));
      UPmatrix = (UPmatrix+conj(UPmatrix))/2;
      UPmse(i) = trace(UPmatrix);     % the upper bound
      
      LBmatrix = n/m*crb+n^2*(m-n)*sigma_e^2*sigma_w^2*((A'*A)\theta_C)*...
       ((A'*A)\theta_C)'/m/(sigma_w+n*sigma_e^2*((theta_C'/(A'*A))*theta_C+(theta_C.'/(A'*A))*conj(theta_C)))/(m*sigma_w+n^2*sigma_e^2*((theta_C'/(A'*A))*theta_C+(theta_C.'/(A'*A))*conj(theta_C)));
      LBmatrix = (LBmatrix+conj(LBmatrix))/2;
      LBmse(i) = trace(LBmatrix);     % the lower bound
      
      mse_uncompressed(i) = mean(mmse_uncompressed);        % the averaged MSE before compression
      mse_compressed(i) = mean(mmse_compressed);      % the averaged MSE after compression
      
      J(i) = trace(crb)/CRB(i);             % the asymptotic relative efficiency ARE
      J_lower(i) =  trace(crb)/UPmse(i);       % the lower bound of ARE
      J_upper(i) =  trace(crb)/LBmse(i);        % the upper bound of ARE
      J_mse(i) = mse_uncompressed(i)/mse_compressed(i);       
end

%% Plot
alw = 0.75;    % AxesLineWidth
fsz = 10;      % Fontsize
lw = 1.5;      % LineWidth
msz = 8;       % MarkerSize
set(gca, 'FontSize', fsz, 'LineWidth', alw);      %<- Set properties
figure(1);
semilogx(sigma_e_sam,J,'-b*',sigma_e_sam,J_lower,':rs',sigma_e_sam,J_upper,'-.mo',sigma_e_sam,J_mse,':k+',sigma_e_sam,m/n*ones(size(sigma_e_sam)),'-.k',sigma_e_sam,(m-p)/(n-p)*ones(size(sigma_e_sam)),':m+','LineWidth',lw,'MarkerSize',msz);
xlabel('\sigma_e^2');
legend('\gamma','\gamma_l','\gamma_u','\gamma_{\rm ML}','m/n','(m-p)/(n-p)');
ylabel('\gamma');

