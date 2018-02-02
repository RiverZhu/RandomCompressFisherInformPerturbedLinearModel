function [ theta_hat ] = gradient_descent(theta_init,A,y,sigma_e,C,sigma_v)
% SUMMARY:
% given measurements: y = (A+EC)*theta + white noise and the initial point,
% this module returns the estimated signals.

% INPUT:
% theta_init: the initial point
% A: the sensing matrix
% y: the measurements
% sigma_e: the strength of perturbation
% C: which can impose some structure on the sensing matrix A
% sigma_v: the variance of the addictive noise

% OUTPUT:
% theta_hat: the estimated signals

[n,~] = size(A);
grad_compute = @(x) -A'*(y-A*x)/(sigma_e*(x'*(C'*C)*x)+sigma_v)+...
               (n-((y-A*x)'*(y-A*x))/(sigma_e*(x'*(C'*C)*x)+sigma_v))*...
               sigma_e*(C'*C)*x/(sigma_e*(x'*(C'*C)*x)+sigma_v);       % the formula of calculating the gradient with respect to conjugate theta 
grad = grad_compute(theta_init);
obj_compute = @(x)(y-A*x)'*(y-A*x)/(sigma_e*(x'*(C'*C)*x)+sigma_v)+n*log(sigma_e*(x'*(C'*C)*x)+sigma_v); % the formula of calculating the objective function 

tol = 1e-5;      % the stopping criterion
step = 1e0;       % the initial step length
iter = 1;       % the initial iteration number
alpha = 0.5; 
beta = 0.5; 
iter_max = 100;      % the maximum iteration number
    while(1)
        if(norm(grad)<tol||iter>iter_max) 
            break;
        end
            while(obj_compute(theta_init-step*grad)>obj_compute(theta_init)-alpha*step*(grad_compute(theta_init)'*grad_compute(theta_init)))
            step = beta*step;
            end        % the backtracking line research to determine step length
    theta_update = theta_init-step*grad;        % update theta_initial
    grad = grad_compute(theta_update);       % update the gradient
    theta_init = theta_update;                      
    iter = iter+1;
    end
theta_hat = theta_init;
end
