function [so, tst, add_inds] = call_pgp_ard(X, Y, Yind, tr_ind, tst_ind, inf_func, tr_ind_not_sel, error_bool, uncertainty_bool)
% Input: X, Y, indicators, training and testing indices, inference function, not
% selected indices (optional), error sampling boolean, uncertainty sampling boolean 
% Output: source results (so, tst (all together)), add_inds (optional) 

% Re-organize modalities 
X = [X(:,1:8) X(:,241:243) X(:,9:240) X(:,244:end)];

% Define the optimization parameters 
Iter = 200; pp.length = -Iter; pp.method = 'LBFGS'; pp.verbosity = 3;

% ARD RBF kernel 
covfunc = {@covSEard_mod2};
likfunc = @likGauss;
D = size(X,2);
hyp.cov = log([ones(D,1); 1; sqrt(1e-5)]);
hyp.cov = log([1; 1; sqrt(1e-2)]);

% Extract the source features and labels
x_s = X(cat(1, tr_ind{:}), :);
y_s = Y(cat(1, tr_ind{:}), :);

% Hyperparameters of the RBF kernel function
hyp.cov(1) = log(median(max(x_s) - min(x_s)));
hyp.cov(2) = log(sqrt(var(y_s(:))));
[~,D]=size(X);
slices = [D-1 D];
slices = [2 11 14 243 609 617 618];
for j=1:numel(slices)
    if j==1
        hyp.cov(j) = log(median(max(x_s(:,1:slices(1))) - min(x_s(:,1:slices(1)))));
    else
        hyp.cov(j) = log(median(max(x_s(:,slices(j-1)+1:slices(j))) - min(x_s(:,slices(j-1)+1:slices(j)))));
    end
end
hyp.cov(j+1) = log(sqrt(var(y_s(:))));
tmp_var = ones(1, size(y_s, 2));
if strcmp(func2str(inf_func), 'infExactW_mod'), hyp.w = log(tmp_var)'; end
hyp.lik = log(sqrt(.1 * var(y_s(:))));

%%%% TRAIN and PREDICT SOURCE MODEL
lk=hyp.cov<=0;
hyp.cov(lk)=100;

% Learn the model parameters from source data
hyp_source = minimize_v2(hyp, @gpoe_train, pp, inf_func, [], covfunc, likfunc, X, Y, tr_ind);
sn2 = exp(2*hyp_source.lik);
ss=1;

%page 19/ alg 2.1 -- compute the posteriors 
[nlZ, ~, post{ss}] = gp(hyp_source, inf_func, [], covfunc, likfunc, x_s, y_s);

tst.m_s = [];
tst.s_s = [];
tst.gt = []; 
tst.yind = [];

% Test the source model
for j = 1:numel(tst_ind)
    
    % extract test data
    xtest  = X(tst_ind{j},:);
    ytstar = Y(tst_ind{j},:);
    ytind = Yind(tst_ind{j},:); 

    K_ts_star = feval(covfunc{:}, hyp_source.cov, x_s, xtest)';
    k_star_star = feval(covfunc{:}, hyp_source.cov, xtest, 'diag');
    V_star{ss} = post{ss}.L'\K_ts_star';

    % mean prediction of the source GP applied to the test data
    mu_t_source  = K_ts_star*post{ss}.alpha;
    % varianc of the source GP applied to the test data
    Sigma_source = k_star_star - sum(V_star{ss}.*V_star{ss})';


    % save them in these variables
    so.m_s{j} = mu_t_source;
    so.s_s{j} = Sigma_source;
    so.gt{j}  = ytstar;
    so.yind{j} = ytind; 
    
    tst.m_s = [tst.m_s; mu_t_source];
    tst.s_s = [tst.s_s; Sigma_source]; 
    tst.gt = [tst.gt; ytstar]; 
    tst.yind = [tst.yind; ytind]; 
    
end

% Note: Same results as above. 
% tst_ind_new{1} = [tst_ind{1}; tst_ind{2}; tst_ind{3}; tst_ind{4}; tst_ind{5}; tst_ind{6}; tst_ind{7}; tst_ind{8}; tst_ind{9}; tst_ind{10}];
% 
% tst2.m_s = [];
% tst2.s_s = [];
% tst2.gt = []; 
% 
% % Test the source model
% for j = 1:numel(tst_ind_new)
%     
%     % extract test data
%     xtest  = X(tst_ind_new{j},:);
%     ytstar = Y(tst_ind_new{j},:);
% 
%     K_ts_star = feval(covfunc{:}, hyp_source.cov, x_s, xtest)';
%     k_star_star = feval(covfunc{:}, hyp_source.cov, xtest, 'diag');
%     V_star{ss} = post{ss}.L'\K_ts_star';
% 
%     % mean prediction of the source GP applied to the test data
%     mu_t_source  = K_ts_star*post{ss}.alpha;
%     % varianc of the source GP applied to the test data
%     Sigma_source = k_star_star - sum(V_star{ss}.*V_star{ss})';
%     
%     tst2.m_s = [tst2.m_s; mu_t_source];
%     tst2.s_s = [tst2.s_s; Sigma_source]; 
%     tst2.gt = [tst2.gt; ytstar]; 
%     
% end

not_sel.m_s = [];
not_sel.s_s = [];
not_sel.gt = []; 

if error_bool 
    % Evaluate on not selected data  
    for j = 1:numel(tr_ind_not_sel)
        
        % extract test data
        xtest  = X(tr_ind_not_sel{j},:);
        ytstar = Y(tr_ind_not_sel{j},:);
        
        K_ts_star = feval(covfunc{:}, hyp_source.cov, x_s, xtest)';
        k_star_star = feval(covfunc{:}, hyp_source.cov, xtest, 'diag');
        V_star{ss} = post{ss}.L'\K_ts_star';
        
        % mean prediction of the source GP applied to the test data
        mu_t_source  = K_ts_star*post{ss}.alpha;
        % varianc of the source GP applied to the test data
        Sigma_source = k_star_star - sum(V_star{ss}.*V_star{ss})';
        
        % save them in these variables
        not_sel.m_s = [not_sel.m_s; mu_t_source];
        not_sel.s_s = [not_sel.s_s; Sigma_source]; 
        not_sel.gt = [not_sel.gt; ytstar];
        
    end
    
    % Get add_inds 
    diff = abs(not_sel.m_s - not_sel.gt); % compute absolute difference 
    max50num = maxk(diff, 50); % get max 50 values 
    max50bool = ismember(diff, max50num); 
    max50inds = find(max50bool); % get max 50 indices (ordered) 
    add_inds = tr_ind_not_sel{:}(max50inds); % get max 50 training indices 

elseif uncertainty_bool 
    % Evaluate on not selected data  
    for j = 1:numel(tr_ind_not_sel)
        
        % extract test data
        xtest  = X(tr_ind_not_sel{j},:);
        ytstar = Y(tr_ind_not_sel{j},:);
        
        K_ts_star = feval(covfunc{:}, hyp_source.cov, x_s, xtest)';
        k_star_star = feval(covfunc{:}, hyp_source.cov, xtest, 'diag');
        V_star{ss} = post{ss}.L'\K_ts_star';
        
        % mean prediction of the source GP applied to the test data
        mu_t_source  = K_ts_star*post{ss}.alpha;
        % varianc of the source GP applied to the test data
        Sigma_source = k_star_star - sum(V_star{ss}.*V_star{ss})';
        
        % save them in these variables
        not_sel.m_s = [not_sel.m_s; mu_t_source];
        not_sel.s_s = [not_sel.s_s; Sigma_source]; 
        not_sel.gt = [not_sel.gt; ytstar];
        
    end
    
    % Get add_inds 
    max50num = maxk(not_sel.s_s, 50); % get max 50 values 
    max50bool = ismember(not_sel.s_s, max50num); 
    max50inds = find(max50bool); % get max 50 indices (ordered) 
    add_inds = tr_ind_not_sel{:}(max50inds); % get max 50 training indices 
    
else 
    add_inds = []; 
end 
end