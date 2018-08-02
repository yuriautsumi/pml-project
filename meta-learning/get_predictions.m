function [predictions, var] = get_predictions(so, j, x_s, x_ad, y_ad, xtest, visit, total_visit, hyp_source, post, covfunc, pgp_bool)
% Input: Source model predictions, patient index, training data, adaptation data, 
% testing data, visit number, total number of vists, parameters from source model,
% covariance function, pGP boolean (true if pGP) 
% Output: filtered predictions and variance for next n steps 

ss = 1; 

% If sGP: 
if ~pgp_bool 
    % Get predictions from sGP model 
    predictions = so.m_s{j}(visit, 1:end)'; 
    var = ones(size(predictions))*so.s_s{j}(visit); 
% Otherwise, if pGP: 
else 
    sn2 = exp(2*hyp_source.lik);
    
    K_ts_star = feval(covfunc{:}, hyp_source.cov, x_s, xtest)';
    k_star_star = feval(covfunc{:}, hyp_source.cov, xtest, 'diag');
    V_star{ss} = post{ss}.L'\K_ts_star';
    
%     mu_t_source  = K_ts_star*post{ss}.alpha; 
%     Sigma_source = k_star_star - sum(V_star{ss}.*V_star{ss})'; 
%     Note: Above evaluates to be same as so.m_s & so.s_s. 
    
    K_ts = feval(covfunc{:}, hyp_source.cov, x_s, x_ad)'; % xt training target domain
    K_tt = feval(covfunc{:}, hyp_source.cov, x_ad);
    V = post{ss}.L'\K_ts';

    mu_t = K_ts*post{ss}.alpha;
    C_t = K_tt - V'*V + sn2*eye(size(K_tt));

    L_adapt = jitterChol(C_t);
    alpha_adapt = solve_chol(L_adapt,(y_ad - mu_t));
    K_t_star = feval(covfunc{:}, hyp_source.cov, x_ad, xtest)';

    C_t_star = K_t_star - V_star{ss}'*V;
    V_adapt = L_adapt'\C_t_star'; 
    
    % mean prediction of the adapted GP applied to the test data
    m_ad = so.m_s{j} + C_t_star*alpha_adapt;
    predictions = m_ad(visit, 1:end)'; 
    % variance of the adapted GP applied to the test data
    s_ad = so.s_s{j} - sum(V_adapt.*V_adapt)';
    var = ones(size(predictions))*s_ad(visit); 
end 

% Filter results as necessary 
diff = min(total_visit-visit, 4); 
predictions = predictions(1:diff); 
var = var(1:diff); 
end 