% Experimenting with active learning 
% Loop over 5 randseeds 
% Test multiple thresholds for Error & Uncertainty Sampling 
% Settings: sGP, Error Sampling, Uncertainty Sampling, Random Sampling 

disp('----- Active Learning Multi-Seed Experiment: Testing Thresholds -----') 
disp('Preparing Data...') 

% Declare variables 
inf_func = @infExactW_mod; 
kernels = {@call_pgp_rbf_threshold, @call_pgp_ard_threshold};
kern_id = ['rbf'; 'ard']; 
error_threshold = [1, 2, 3, 4]; 
var_threshold = [5, 10, 15, 20]; 

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
for f = 1:2 
    
    disp(['----- Fold ' num2str(f) ' -----']) 
    
    % Extract data for fold 
    lid = circshift([1:100],(1-f)*50); 
    
    te_set = lid(1:50); 
    tr_set = lid(51:end); 
    
    % Indices of the test data (the first 50 subjects) that will also be the adaptation data as t->t+1->t+2 etc.
    for i=1:numel(te_set)
        tst_ind{i}=find(data(:,1)==rid(te_set(i)));
    end
    
    % Indices of the source data
    tr_ind{1}=[];
    for i=1:numel(tr_set)
        tr_ind{1}=[tr_ind{1};find(data(:,1)==rid(tr_set(i)))];
    end
    
    % For each error threshold 
    for e = error_threshold
        % For each variance threshold 
        for v = var_threshold 
            % For RBF kernel and ARD kernel: 
            for kern = 1:numel(kernels) 
                % How to call pgp fxn: kernels{kern}(inputs)
                % For each random seed: 
                for s = [5, 6, 7, 8, 9]
                    
                    disp(['----- Error ' num2str(e) ' | Variance ' num2str(v) ' | Kernel ' kern_id(kern, 1:end) ' | Seed ' num2str(s) ' -----']) 
                    
                    % Set seed 
                    rng(s); 

                    disp('----- Model 1: sGP -----') 
                    % MODEL 1: sGP 
                    % Call pgp function 
                    [so, tst, ~] = kernels{kern}(X, Y, Yind, tr_ind, tst_ind, inf_func, [], false, false, e, v); 

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
                    csvwrite([currentDir '/al_exp_seed_threshold_results/f' num2str(f) '_s' num2str(s) '_m1_error_' kern_id(kern, :) '_e' num2str(e) '_v' num2str(v) '.csv'], m1_errors);

                    disp('----- Model 2: Error Sampling -----') 
                    % MODEL 2: Error Sampling 
                    % Start out with 100 training data 
                    tr_ind_random = tr_ind{:}(randperm(length(tr_ind{:}))); % Randomly sorted training indices 

                    tr_ind_sel{1} = tr_ind_random(1:100); 
                    tr_ind_not_sel{1} = tr_ind_random(101:end); 
                    
                    num_data = [100];

                    loop = 0; 
                    m2_iter_errors = []; % Errors for each iteration 
                    add_inds = [];

                    % While first loop around or add_inds is not empty: 
                    while loop == 0 | ~isempty(add_inds)
                        % If it is not the first loop around: 
                        if loop ~= 0 
                            % Update x_s_sel with add_inds from previous iteration 
                            tr_ind_sel{1} = [tr_ind_sel{:}; add_inds]; 
                            tr_ind_not_sel{1} = setdiff(tr_ind_not_sel{:}, add_inds);
                            
                            num_data = [num_data; num_data(end)+length(add_inds)]; 
                        end 
                        
                        % Call pgp function 
                        [so, tst, add_inds] = kernels{kern}(X, Y, Yind, tr_ind_sel, tst_ind, inf_func, tr_ind_not_sel, true, false, e, v); 

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

                    % Append average error of model 1 as last value &
                    % assemble error array 
                    m2_iter_errors = [m2_iter_errors; avg_m1_error]; 
                    num_data = [num_data; length(tr_ind{:})]; 
                    m2_errors = [num_data m2_iter_errors]; 

                    % Save errors 
                    csvwrite([currentDir '/al_exp_seed_threshold_results/f' num2str(f) '_s' num2str(s) '_m2_error_' kern_id(kern, :) '_e' num2str(e) '_v' num2str(v) '.csv'], m2_errors);

                    disp('----- Model 3: Uncertainty (Variance) Sampling -----') 
                    % MODEL 3: Uncertainty (Variance) Sampling 
                    % Start out with 100 training data 
                    tr_ind_random = tr_ind{:}(randperm(length(tr_ind{:}))); % Randomly sorted training indices 

                    tr_ind_sel{1} = tr_ind_random(1:100); 
                    tr_ind_not_sel{1} = tr_ind_random(101:end); 
                    
                    num_data = [100];

                    loop = 0; 
                    m3_iter_errors = []; % Errors for each iteration 
                    add_inds = [];

                    % While first loop around or add_inds is not empty: 
                    while loop == 0 | ~isempty(add_inds)
                        % If it is not the first loop around: 
                        if loop ~= 0
                            % Update x_s_sel with add_inds from previous iteration 
                            tr_ind_sel{1} = [tr_ind_sel{:}; add_inds]; 
                            tr_ind_not_sel{1} = setdiff(tr_ind_not_sel{:}, add_inds);
                            
                            num_data = [num_data; num_data(end)+length(add_inds)];
                        end 

                        % Call pgp function 
                        [so, tst, add_inds] = kernels{kern}(X, Y, Yind, tr_ind_sel, tst_ind, inf_func, tr_ind_not_sel, false, true, e, v); 

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

                        % Evaluate average error and append to m3_iter_errors 
                        avg_m3_error = mean(all_tst_error); 
                        m3_iter_errors = [m3_iter_errors; avg_m3_error]; 

                        loop = loop + 1; 
                    end 

                    % Append average error of model 1 as last value 
                    m3_iter_errors = [m3_iter_errors; avg_m1_error]; 
                    num_data = [num_data; length(tr_ind{:})]; 
                    m3_errors = [num_data m3_iter_errors];

                    % Save errors 
                    csvwrite([currentDir '/al_exp_seed_threshold_results/f' num2str(f) '_s' num2str(s) '_m3_error_' kern_id(kern, :) '_e' num2str(e) '_v' num2str(v) '.csv'], m3_errors);            

                    disp('----- Model 4: Random Sampling -----') 
                    % MODEL 4: Random Sampling 
                    % Start out with 100 training data 
                    tr_ind_random = tr_ind{:}(randperm(length(tr_ind{:}))); % Randomly sorted training indices 

                    tr_ind_sel{1} = tr_ind_random(1:100); 
                    tr_ind_not_sel{1} = tr_ind_random(101:end); 

                    loop = 0; 
                    m4_iter_errors = []; 

                    % While tr_ind_not_sel{1} has at least 50 elements: 
                    while length(tr_ind_not_sel{1}) > 50
                        % If it is not the first loop around: 
                        if loop ~= 0
                            % Update x_s_sel with add_inds (randomly selected 50 points) 
                            tr_ind_not_sel_random = tr_ind_not_sel{:}(randperm(length(tr_ind_not_sel{:}))); % Randomly sorted not selected training indices 
                            add_inds = tr_ind_not_sel_random(1:50); % Randomly select 50 points 

                            tr_ind_sel{1} = [tr_ind_sel{:}; add_inds]; 
                            tr_ind_not_sel{1} = setdiff(tr_ind_not_sel{:}, add_inds);
                        end 

                        % Call pgp function 
                        [so, tst, ~] = kernels{kern}(X, Y, Yind, tr_ind_sel, tst_ind, inf_func, [], false, false, e, v); 

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

                        % Evaluate average error and append to m4_iter_errors 
                        avg_m4_error = mean(all_tst_error); 
                        m4_iter_errors = [m4_iter_errors; avg_m4_error]; 

                        loop = loop + 1; 
                    end 

                    % Append average error of model 1 as last value 
                    m4_iter_errors = [m4_iter_errors; avg_m1_error]; 

                    % Save errors 
                    csvwrite([currentDir '/al_exp_seed_threshold_results/f' num2str(f) '_s' num2str(s) '_m4_error_' kern_id(kern, :) '_e' num2str(e) '_v' num2str(v) '.csv'], m4_iter_errors);

                end 
            end 
        end 
    end 
end 