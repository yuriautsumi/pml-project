% Runs 4-Step Horizon Meta-Learning Algorithm 

disp('----- 4 Time-Step Meta Learning Experiment -----') 
disp('Preparing Data...') 

% Declare variables 
threshold_vals = [0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]; 
inf_func = @infExactW_mod; 
kernels = {@call_sgp_rbf, @call_sgp_ard};
cov_funcs = {@covSEiso, @covSEard_mod2}; 
kern_id = ['rbf'; 'ard']; 
% kernels = {@call_sgp_rbf};
% cov_funcs = {@covSEiso}; 
% kern_id = ['rbf']; 
% kernels = {@call_sgp_ard};
% cov_funcs = {@covSEard_mod2}; 
% kern_id = ['ard']; 

% Get data 
currentDir = pwd; 
data = csvread([currentDir '/adni_adas13_100_fl1_l4.csv']); 

rid=unique(data(:,1));

X    = data(:,2:end-8); % the first column is the ID so we remove it; the rest should be used as here
Y    = data(:,end-7:end-4); 
Yind = data(:,end-3:end);

% Initialize output array for each threshold value 
% Each cell will end up as 2100x1 double 
for t = 1:numel(threshold_vals)
    s_output_rbf{t} = []; 
end 

for t = 1:numel(threshold_vals)
    ml_output_rbf{t} = []; 
end 

for t = 1:numel(threshold_vals)
    s_output_ard{t} = []; 
end 

for t = 1:numel(threshold_vals)
    ml_output_ard{t} = []; 
end 

% Initialize error array for each threshold value 
% Each cell will end up as 100x1 double 
for t = 1:numel(threshold_vals)
    s_error_rbf{t} = []; 
end 

for t = 1:numel(threshold_vals)
    ml_error_rbf{t} = []; 
end 

for t = 1:numel(threshold_vals)
    s_error_ard{t} = []; 
end 

for t = 1:numel(threshold_vals)
    ml_error_ard{t} = []; 
end 

% Initialize requests array for each threshold value 
% Each cell will end up as 100x1 double 
for t = 1:numel(threshold_vals)
    ml_requests_rbf{t} = []; 
    ml_req_vec_rbf{t} = [];
    ml_ad_vec_rbf{t} = [];
end 

for t = 1:numel(threshold_vals)
    ml_requests_ard{t} = []; 
    ml_req_vec_ard{t} = [];
    ml_ad_vec_ard{t} = []; 
end 

% For each fold: 
for f = 1:10
    
    disp(['----- Fold ' num2str(f) ' -----']) 
    
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
        
        % Build & train sGP model (with & without slices) 
        [so_results, hyp_source, post] = kernels{kern}(X, Y, Yind, tr_ind, tst_ind, inf_func); 
        
        % For each threshold value: 
        for t = 1:numel(threshold_vals)
            
            disp(['----- Kernel ' kern_id(kern, 1:end) ' | Threshold ' num2str(threshold_vals(t))])
            
            % Call 4-step horizon meta-learning algorithm 
            [results] = call_ml_4ts(X, Y, tr_ind, tst_ind, so_results, hyp_source, post, cov_funcs(kern), threshold_vals(t)); 
            
            % Process Results 
            % For each test patient: 
            all_m_s = []; 
            all_s_s = []; 
            all_m_ml = []; 
            all_s_ml = []; 
            
            all_s_tst_error = []; 
            all_ml_tst_error = []; 
            
            all_reqs = []; 
            
            all_req_vec = []; 
            all_ad_vec = []; 
            
            for j = 1:numel(tst_ind)
                % Get corresponding predictions, ground truth, indicators 
                pat_m_s = so_results.m_s{j}; 
                pat_s_s = so_results.s_s{j}; 
                pat_m_ml = results.m{j};
                pat_s_ml = results.s{j}; 
                pat_gt = results.gt{j}; 
                pat_yind = results.yind{j}; 
                
                non_zero_inds = find(pat_yind~=0); 
                pat_gt_processed = pat_gt(non_zero_inds); 
                pat_m_s_processed = pat_m_s(non_zero_inds);
                pat_m_ml_processed = pat_m_ml(non_zero_inds); 
                
                source_pat_error = compute_mse(pat_gt_processed, pat_m_s_processed); 
                meta_pat_error = compute_mse(pat_gt_processed, pat_m_ml_processed); 
                
                all_m_s = [all_m_s; pat_m_s]; 
                all_s_s = [all_s_s; pat_s_s]; 
                all_m_ml = [all_m_ml; pat_m_ml]; 
                all_s_ml = [all_s_ml; pat_s_ml]; 
                
                all_s_tst_error = [all_s_tst_error; source_pat_error];
                all_ml_tst_error = [all_ml_tst_error; meta_pat_error]; 
                
                all_reqs = [all_reqs; sum(results.req_visits{j})]; 
                
                all_req_vec = [all_req_vec; results.req_visits{j}']; 
                all_ad_vec = [all_ad_vec; results.ad_visits{j}']; 
            end 
            
            if kern == 1 
                s_output_rbf{t} = [s_output_rbf{t}; all_m_s all_s_s]; 
                ml_output_rbf{t} = [ml_output_rbf{t}; all_m_ml all_s_ml]; 
                
                s_error_rbf{t} = [s_error_rbf{t}; all_s_tst_error]; 
                ml_error_rbf{t} = [ml_error_rbf{t}; all_ml_tst_error];                 
                
                ml_requests_rbf{t} = [ml_requests_rbf{t}; all_reqs]; 
                
                ml_req_vec_rbf{t} = [ml_req_vec_rbf{t}; all_req_vec]; 
                ml_ad_vec_rbf{t} = [ml_ad_vec_rbf{t}; all_ad_vec]; 
            else 
                s_output_ard{t} = [s_output_ard{t}; all_m_s all_s_s]; 
                ml_output_ard{t} = [ml_output_ard{t}; all_m_ml all_s_ml]; 
                
                s_error_ard{t} = [s_error_ard{t}; all_s_tst_error]; 
                ml_error_ard{t} = [ml_error_ard{t}; all_ml_tst_error];                 
                
                ml_requests_ard{t} = [ml_requests_ard{t}; all_reqs]; 
                
                ml_req_vec_ard{t} = [ml_req_vec_ard{t}; all_req_vec]; 
                ml_ad_vec_ard{t} = [ml_ad_vec_ard{t}; all_ad_vec];                 
            end 
            
        end 
    end 
end 

disp('Finished script') 
% Output results, errors, and requests to CSV 
% Get group indices 
group_class = csvread([currentDir '/patient_classification.csv']); 
one_ind = find(group_class(1:end, 2) == 1); 
two_ind = find(group_class(1:end, 2) == 2); 
three_ind = find(group_class(1:end, 2) == 3); 

% s_output_rbf
% For each threshold value: 
for t = 1:numel(threshold_vals)
    
    save_results(t, threshold_vals, s_error_rbf, s_output_rbf, [], [], [], rid, one_ind, two_ind, three_ind, currentDir, true, true); 
    
end 

% ml_output_rbf
for t = 1:numel(threshold_vals)
    
    save_results(t, threshold_vals, ml_error_rbf, ml_output_rbf, ml_requests_rbf, ml_req_vec_rbf, ml_ad_vec_rbf, rid, one_ind, two_ind, three_ind, currentDir, false, true); 
    
end 

% s_output_ard
for t = 1:numel(threshold_vals)
    
    save_results(t, threshold_vals, s_error_ard, s_output_ard, [], [], [], rid, one_ind, two_ind, three_ind, currentDir, true, false); 
    
end 

% ml_output_ard
for t = 1:numel(threshold_vals)
    
    save_results(t, threshold_vals, ml_error_ard, ml_output_ard, ml_requests_ard, ml_req_vec_ard, ml_ad_vec_ard, rid, one_ind, two_ind, three_ind, currentDir, false, false); 
    
end 