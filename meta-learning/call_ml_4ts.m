function [results] = call_ml_4ts(X, Y, tr_ind, tst_ind, so_results, hyp_source, post, covfunc, threshold)
% Summary: Gives results based on 4-step horizon meta-learning algorithm. 
% Call once per threshold value per fold. 
% Input: feature array, label vector, training indices, test indices, sGP
% results, sGP parameters, covariance function, threshold value
% Output: mean, variance, ground truths, indicators, requested visits,
% visits used for adaptation data 

% Extract data 
x_s = X(cat(1, tr_ind{:}), :);
y_s = Y(cat(1, tr_ind{:}), :);

% For each test patient: 
for j = 1:numel(tst_ind) 
    v = 1; % total visit count 
    
    % Get corresponding predictions, ground truth, indicators, test data 
%     pat_ms = so_results.m_s{j}; 
%     pat_ss = so_results.s_s{j}; 
    pat_X = so_results.x{j}; 
    pat_Y = so_results.gt{j}; 
    pat_yind = so_results.yind{j}; 
    xtest  = X(tst_ind{j},:);
    
    req_visits = []; % list of requested visits 
    ad_visits = []; % list of visits used for adaptation data 
    
    pat_pgp_m = nan(length(pat_Y), 1); % initialize results array 
    pat_pgp_s = nan(length(pat_Y), 1); % initialize results array 
    
    x_ad = []; % initialize adaptation data array 
    y_ad = []; % initialize adaptation data array 
    
    pgp_bool = false; % whether pGP model is usable 
    req_bool = true; % whether request is made for visit v 
    
    % While results array is not full: 
    while sum(~isnan(pat_pgp_m)) ~= length(pat_Y) 
        % If label is requested: 
        if req_bool 
            % Get (xv, yv) 
            req_visits = [req_visits; v]; 
            xyv = pat_X(v, 1:end); 
            xv = pat_X(v, 1:end-1); 
            yv = pat_X(v, end:end); 
            
            % Save requested/observed values 
            xy_req{v} = xyv; 
            x_req{v} = xv; 
            y_req{v} = yv; 
            
%             assert((sum(~isnan(pat_pgp_m))+1) == v) 
            
            % Keep yv 
            pat_pgp_m(v) = yv; 
            pat_pgp_s(v) = 0; 
        end 
        
        % Check if pGP can be updated 
        [updatable, updatable_visit] = check_pgp_update(req_visits, ad_visits, pat_pgp_m);   
        
        % Update as necessary 
        [x_ad, y_ad, ad_visits] = update_pgp(updatable, updatable_visit, ad_visits, xy_req, pat_pgp_m, x_ad, y_ad); 
        
        % If updated, set pgp boolean to true 
        if updatable pgp_bool = true; end 
        
        % Get relevant predictions 
        [y_pred, s_pred] = get_predictions(so_results, j, x_s, x_ad, y_ad, xtest, v, length(pat_Y), hyp_source, post, covfunc, pgp_bool); 
        
        v0 = v; 
        skip_next = false; 
        
        % First go through all good predictions  
        for p = 1:length(y_pred) 
            % Increment visit count 
            v = v+1; 
            
            % Get variables 
            yhat = y_pred(p); 
            shat = s_pred(p); 
            gtv = pat_X(v, end:end); % ground truth value for this visit
            
            % If error < threshold: 
            if ~eval_error(yhat, gtv, threshold)
                req_bool = false; 
                
%                 assert((sum(~isnan(pat_pgp_m))+1) == v) 

                % Keep yhat  
                pat_pgp_m(v) = yhat; 
                pat_pgp_s(v) = shat; 
                
                % Check if updatable 
                [updatable, updatable_visit] = check_pgp_update(req_visits, ad_visits, pat_pgp_m); 
                if updatable 
                    skip_next = true; 
                    break 
                end 
            end 
        end 
        
        % Go through rest of predictions if ~skip_next 
        if ~skip_next 
            v = v0; % Reset visits  
            for p = 1:length(y_pred)   
                % Increment visit count 
                v = v+1; 

                % Get variables 
                yhat = y_pred(p); 
                shat = s_pred(p); 
                gtv = pat_X(v, end:end); % ground truth value for this visit
                
                % If error > threshold: 
                if eval_error(yhat, gtv, threshold)
                    req_bool = true; 
                    break 
                % Otherwise, if it is the last of the prediction set and not
                % the 21st visit 
                elseif find(pat_pgp_m == yhat) == length(pat_pgp_m) && v ~= length(pat_Y)
                    req_bool = true; 
                    break 
                end 
            end 
        end 
    end 
    
    % Save results 
    results.m{j} = pat_pgp_m; 
    results.s{j} = pat_pgp_s; 
    results.gt{j} = pat_X(1:end, end:end);  
    results.yind{j} = pat_yind(1:end, 1:1); 
    
    initial_array1 = zeros(length(pat_Y), 1); 
    initial_array2 = zeros(length(pat_Y), 1); 
    initial_array1(req_visits) = 1; 
    initial_array2(ad_visits) = 1; 
    results.req_visits{j} = initial_array1;
    results.ad_visits{j} = initial_array2;
end 
end 