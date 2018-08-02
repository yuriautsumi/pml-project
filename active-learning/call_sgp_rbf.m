function [so, hyp_source, post] = call_sgp_rbf(X, Y, Yind, tr_ind, tst_ind, inf_func)
% Inputs: feature array, label vector, training indices, test indices,
% inference function 

% Define optimization parameters 
Iter = 200; pp.length = -Iter; pp.method = 'LBFGS'; pp.verbosity = 3;

% Standard RBF kernel ISO
covfunc = {@covSEiso};
likfunc = @likGauss;

% Extract source features and labels
x_s = X(cat(1, tr_ind{:}), :);
y_s = Y(cat(1, tr_ind{:}), :);

% Initialize hyperparameters 
hyp.cov(1) = log(median(max(x_s) - min(x_s)));
hyp.cov(2) = log(sqrt(var(y_s(:))));
tmp_var = ones(1, size(y_s, 2));
if strcmp(func2str(inf_func), 'infExactW_mod'), hyp.w = log(tmp_var)'; end
hyp.lik = log(sqrt(.1 * var(y_s(:))));

lk=hyp.cov<=10^-5;
hyp.cov(lk)=10^-5;

%%%% TRAIN and PREDICT SOURCE MODEL
% Learn model parameters from source data
hyp_source = minimize_v2(hyp, @gpoe_train, pp, inf_func, [], covfunc, likfunc, X, Y, tr_ind);
ss=1;

lk=hyp_source.cov<=10^-5;
hyp_source.cov(lk)=10^-5;

% page 19/ alg 2.1 -- Compute posteriors 
try 
    [~, ~, post{ss}] = gp(hyp_source, inf_func, [], covfunc, likfunc, x_s, y_s);
catch 
    pause(25);
end 

% Test the source model
for j = 1:numel(tst_ind)
    
    % Extract test data
    xtest  = X(tst_ind{j},:);
    ytstar = Y(tst_ind{j},:);
    ytind = Yind(tst_ind{j},:);
    
    K_ts_star = feval(covfunc{:}, hyp_source.cov, x_s, xtest)';
    k_star_star = feval(covfunc{:}, hyp_source.cov, xtest, 'diag');
    V_star{ss} = post{ss}.L'\K_ts_star';
    
    % Mean prediction of the source GP applied to the test data
    mu_t_source  = K_ts_star*post{ss}.alpha;
    % Variance of the source GP applied to the test data
    Sigma_source = k_star_star - sum(V_star{ss}.*V_star{ss})';
    
    % Save 
    so.m_s{j} = mu_t_source;
    so.s_s{j} = Sigma_source;
    so.x{j} = xtest; 
    so.gt{j}  = ytstar;
    so.yind{j} = ytind; 
    
end
end 