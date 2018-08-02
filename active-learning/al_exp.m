% Experimenting with active learning 

% Set seed 
rng(0);

% Declare variables 
inf_func = @infExactW_mod; 
kernels = {@call_pgp_rbf, @call_pgp_ard};
kern_id = ['rbf'; 'ard']; 

% Get data (X, Y, ids) 
currentDir = pwd; 
data = csvread([currentDir '/adni_adas13_100_fl1_l4.csv']); 

rid=unique(data(:,1));

X    = data(:,2:end-8); % the first column is the ID so we remove it; the rest should be used as here
Y    = data(:,end-7:end-7); 
Yind = data(:,end-3:end-3); 
% Y    = data(:,end-7:end-4); 
% Yind = data(:,end-3:end); 

tind=[]; % 1 to 2100 column vector 
for j=1:numel(rid) 
    idd = find(data(:,1)==rid(j)); 
    tind = [tind;idd]; 
end

% For each fold: 
for f = 1:10 
    % Extract data for fold 
    lid = circshift([1:100],(1-f)*10); 
    
    te_set = lid(1:10); 
    tr_set = lid(11:end); 
    
    % Indices of the test data (the first 10 subjects) that will also be the adaptation data as t->t+1->t+2 etc.
    for i=1:numel(te_set)
        tst_ind{i}=find(data(:,1)==rid(te_set(i)));
    end
    
    % Indices of the source data
    tr_ind{1}=[];
    for i=1:numel(tr_set)
        tr_ind{1}=[tr_ind{1};find(data(:,1)==rid(tr_set(i)))];
    end
    
    % For RBF kernel and ARD kernel 
    for kern = 1:numel(kernels) 
        % How to call pgp fxn: kernels{kern}(inputs)
        
        % MODEL 1: sGP 
        % Call pgp function 
        [so, tst, ~] = kernels{kern}(X, Y, Yind, tr_ind, tst_ind, inf_func, [], false, false); 
        
        % Evaluate error for each patient 
        all_tst_error = [];
        for pat = 1:numel(tst_ind)
            % Get corresponding predictions, ground truth, indicators 
            pat_m_s = so.m_s{pat}; 
            pat_gt = so.gt{pat}; 
            pat_ind = so.yind{pat}; 
            
            non_zero_inds = find(pat_ind~=0); 
            pat_gt_processed = pat_gt(non_zero_inds); 
            pat_m_s_processed = pat_m_s(non_zero_inds); 
            
            source_pat_error = compute_mse(pat_gt_processed, pat_m_s_processed); 
            
            all_tst_error = [all_tst_error; source_pat_error]; 
        end 
        
        avg_m1_error = mean(all_tst_error); 
        m1_errors = [rid(te_set) all_tst_error]; 
        
        % Save errors 
        csvwrite([currentDir '/al_exp_results/f' num2str(f) '_m1_error_' kern_id(kern, :) '.csv'], m1_errors);
        
        % MODEL 2: Uncertainty (Error) Sampling 
        % Start out with 100 training data 
        tr_ind_random = tr_ind{:}(randperm(length(tr_ind{:}))); % Randomly sorted training indices 
        
        tr_ind_sel{1} = tr_ind_random(1:100); 
        tr_ind_not_sel{1} = tr_ind_random(101:end); 
        
        loop = 0; 
        m2_iter_errors = []; % Errors for each iteration 
        
        % While tr_ind_not_sel{1} has at least 50 elements: 
        while length(tr_ind_not_sel{1}) >= 50
            % If it is not the first loop around: 
            if loop ~= 0
                % Update x_s_sel with add_inds from previous iteration 
                tr_ind_sel{1} = [tr_ind_sel{:}; add_inds]; 
                tr_ind_not_sel{1} = setdiff(tr_ind_not_sel{:}, add_inds);
            end 
            
            % Call pgp function 
            [so, tst, add_inds] = kernels{kern}(X, Y, Yind, tr_ind_sel, tst_ind, inf_func, tr_ind_not_sel, true, false); 
            
            % Evaluate error for each patient 
            all_tst_error = [];
            for pat = 1:numel(tst_ind)
                % Get corresponding predictions, ground truth, indicators 
                pat_m_s = so.m_s{pat}; 
                pat_gt = so.gt{pat}; 
                pat_ind = so.yind{pat}; 

                non_zero_inds = find(pat_ind~=0); 
                pat_gt_processed = pat_gt(non_zero_inds); 
                pat_m_s_processed = pat_m_s(non_zero_inds); 

                source_pat_error = compute_mse(pat_gt_processed, pat_m_s_processed); 

                all_tst_error = [all_tst_error; source_pat_error]; 
            end 
            
            % Evaluate average error and append to m2_iter_errors 
            avg_m2_error = mean(all_tst_error); 
            m2_iter_errors = [m2_iter_errors; avg_m2_error]; 
            
            loop = loop + 1; 
        end 
            
        % Append average error of model 1 as last value 
        m2_iter_errors = [m2_iter_errors; avg_m1_error]; 
        
        % Save errors 
        csvwrite([currentDir '/al_exp_results/f' num2str(f) '_m2_error_' kern_id(kern, :) '.csv'], m2_iter_errors);
        
        % MODEL 3: Random Sampling 
        % Start out with 100 training data 
        tr_ind_random = tr_ind{:}(randperm(length(tr_ind{:}))); % Randomly sorted training indices 
        
        tr_ind_sel{1} = tr_ind_random(1:100); 
        tr_ind_not_sel{1} = tr_ind_random(101:end); 
        
        loop = 0; 
        m3_iter_errors = []; 
        
        % While tr_ind_not_sel{1} has at least 50 elements: 
        while length(tr_ind_not_sel{1}) >= 50
            % If it is not the first loop around: 
            if loop ~= 0
                % Update x_s_sel with add_inds (randomly selected 50 points) 
                tr_ind_not_sel_random = tr_ind_not_sel{:}(randperm(length(tr_ind_not_sel{:}))); % Randomly sorted not selected training indices 
                add_inds = tr_ind_not_sel_random(1:50); % Randomly select 50 points 
                
                tr_ind_sel{1} = [tr_ind_sel{:}; add_inds]; 
                tr_ind_not_sel{1} = setdiff(tr_ind_not_sel{:}, add_inds);
            end 
            
            % Call pgp function 
            [so, tst, ~] = kernels{kern}(X, Y, Yind, tr_ind_sel, tst_ind, inf_func, [], false, false); 
            
            % Evaluate error for each patient 
            all_tst_error = [];
            for pat = 1:numel(tst_ind)
                % Get corresponding predictions, ground truth, indicators 
                pat_m_s = so.m_s{pat}; 
                pat_gt = so.gt{pat}; 
                pat_ind = so.yind{pat}; 

                non_zero_inds = find(pat_ind~=0); 
                pat_gt_processed = pat_gt(non_zero_inds); 
                pat_m_s_processed = pat_m_s(non_zero_inds); 

                source_pat_error = compute_mse(pat_gt_processed, pat_m_s_processed); 

                all_tst_error = [all_tst_error; source_pat_error]; 
            end 
            
            % Evaluate average error and append to m2_iter_errors 
            avg_m3_error = mean(all_tst_error); 
            m3_iter_errors = [m3_iter_errors; avg_m3_error]; 
            
            loop = loop + 1; 
        end 
            
        % Append average error of model 1 as last value 
        m3_iter_errors = [m3_iter_errors; avg_m1_error]; 
        
        % Save errors 
        csvwrite([currentDir '/al_exp_results/f' num2str(f) '_m3_error_' kern_id(kern, :) '.csv'], m3_iter_errors);
        
    end 
end 