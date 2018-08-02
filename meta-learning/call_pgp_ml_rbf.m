function [results] = call_pgp_ml_rbf(X, Y, Yind, tr_ind, tst_ind, inf_func, threshold_vals)
% Inputs: feature array, label vector, training indices, test indices,
% inference function, threshold value array 

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

%%%% TRAIN and PREDICT SOURCE MODEL
% Learn model parameters from source data
hyp_source = minimize_v2(hyp, @gpoe_train, pp, inf_func, [], covfunc, likfunc, X, Y, tr_ind);
sn2 = exp(2*hyp_source.lik);
ss=1;

% page 19/ alg 2.1 -- Compute posteriors 
[nlZ, ~, post{ss}] = gp(hyp_source, inf_func, [], covfunc, likfunc, x_s, y_s);

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
    so.gt{j}  = ytstar;
    so.yind{j} = ytind; 
    
end

%%% END OF SOURCE

%%% BEGIN ADAPTATION %%%

clear xtest ytstar

% iterate over all threshold values 
for t = 1:numel(threshold_vals) 
    threshold = threshold_vals(t);
    for j = 1:numel(tst_ind) % for subjects 1->N
        
        m_so=[]; s_so=[]; m_ad=[]; s_ad=[]; g_t =[]; req_vec = []; 

        % For each new subject, clear adaptation data 
        clear x_ad y_ad 
        
        x_ad = [];
        y_ad = [];
        
        req = 0; 
        
        for kk = 1:numel(tst_ind{j}) % for visit 1->L of the j-th subject
            
            if kk==1 
               % if the first visit, use the prediction by the source model
               m_so=[m_so;so.m_s{j}(1,:)];
               s_so=[s_so;so.s_s{j}(1,:)];
               m_ad=[m_ad;so.m_s{j}(1,:)];
               s_ad=[s_ad;so.s_s{j}(1,:)];
               g_t=[g_t;so.gt{j}(1,:)];
               req_vec = [req_vec; 0]; 
               
               bool = eval_error(so.m_s{j}(1,:), so.gt{j}(1,:), threshold); 
               
               continue        
            end
            
            % If error is greater than threshold: 
            if eval_error(so.m_s{j}(kk,:), so.gt{j}(kk,:), threshold) 
                % increment requests by 1
                req = req+1; 
                
                % update adaptation data & proceed with adaptation 
                x_ad    = [x_ad; X(tst_ind{j}(kk-1:kk-1),:)]; 
                y_ad    = [y_ad; Y(tst_ind{j}(kk-1:kk-1),:)]; 
                
                % test data for subject j
                xtest = X(tst_ind{j}(kk),:);
                ytstar= Y(tst_ind{j}(kk),:);
                
                % accumulate the ground truth labels
                g_t = [g_t;ytstar];
                
                % load the source data again to compute new kernels 
                x_s = X(tr_ind{ss},:);
                y_s = Y(tr_ind{ss},:);
                
                %%%%%%% the source model predictions
                K_ts_star = feval(covfunc{:}, hyp_source.cov, x_s, xtest)';
                k_star_star = feval(covfunc{:}, hyp_source.cov, xtest, 'diag');
                V_star{ss} = post{ss}.L'\K_ts_star';
                
                mu_t_source  = K_ts_star*post{ss}.alpha;
                Sigma_source = k_star_star - sum(V_star{ss}.*V_star{ss})';
                
                %%%%%%%
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
                mu_adapt = mu_t_source + C_t_star*alpha_adapt;
                % variance of the adapted GP applied to the test data
                Sigma_adapt = Sigma_source - sum(V_adapt.*V_adapt)';
                
                m_so=[m_so;mu_t_source];
                s_so=[s_so;Sigma_source];
                m_ad=[m_ad;mu_adapt];
                s_ad=[s_ad;Sigma_adapt];
                req_vec = [req_vec; 1];
                
            % Otherwise, if there is no adaptation data: 
            elseif isempty(x_ad) & isempty(y_ad)
                % use prediction by the source model 
                m_so=[m_so;so.m_s{j}(kk,:)];
                s_so=[s_so;so.s_s{j}(kk,:)];
                m_ad=[m_ad;so.m_s{j}(kk,:)];
                s_ad=[s_ad;so.s_s{j}(kk,:)];
                g_t=[g_t;so.gt{j}(kk,:)];
                req_vec = [req_vec; 0]; 
                
            % Otherwise, if error is not greater than threshold &
            % adaptation data exists: 
            else 
                % don't update adaptation data & proceed with adaptation 
                % test data for subject j
                xtest = X(tst_ind{j}(kk),:);
                ytstar= Y(tst_ind{j}(kk),:);
                
                % accumulate the ground truth labels
                g_t = [g_t;ytstar];
                
                % load the source data again to compute new kernels 
                x_s = X(tr_ind{ss},:);
                y_s = Y(tr_ind{ss},:);
                
                %%%%%%% the source model predictions
                K_ts_star = feval(covfunc{:}, hyp_source.cov, x_s, xtest)';
                k_star_star = feval(covfunc{:}, hyp_source.cov, xtest, 'diag');
                V_star{ss} = post{ss}.L'\K_ts_star';

                mu_t_source  = K_ts_star*post{ss}.alpha;
                Sigma_source = k_star_star - sum(V_star{ss}.*V_star{ss})';
                
                %%%%%%%
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
                mu_adapt = mu_t_source + C_t_star*alpha_adapt;
                % variance of the adapted GP applied to the test data
                Sigma_adapt = Sigma_source - sum(V_adapt.*V_adapt)';

                m_so=[m_so;mu_t_source];
                s_so=[s_so;Sigma_source];
                m_ad=[m_ad;mu_adapt];
                s_ad=[s_ad;Sigma_adapt];
                req_vec = [req_vec; 0]; 
            end
        end 
        % save them in these variables
        ad.m_s{j}  = m_so;
        ad.s_s{j}  = s_so;
        ad.m_ad{j}  = m_ad;
        ad.s_ad{j}  = s_ad;
        ad.gt{j}    = g_t; 
        ad.yind{j} = so.yind{j};
        ad.reqs{j} = req; 
        ad.req_vec{j} = req_vec; 
    end
    results{t} = ad;
end 
end