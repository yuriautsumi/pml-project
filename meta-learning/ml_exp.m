% Experimenting with meta-learning 
% Use updated call_pgp_ml.m 

disp('----- Meta Learning Experiment -----') 
disp('Preparing Data...') 

% Declare variables 
threshold_vals = [0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]; 
inf_func = @infExactW_mod; 
kernels = {@call_pgp_ml_rbf, @call_pgp_ml_ard};
kern_id = ['rbf'; 'ard']; 

% Get data 
currentDir = pwd; 
data = csvread([currentDir '/adni_adas13_100_fl1_l4.csv']); 

rid=unique(data(:,1));

X    = data(:,2:end-8); % the first column is the ID so we remove it; the rest should be used as here
Y    = data(:,end-7:end-7); 
Yind = data(:,end-3:end-3);
% Y    = data(:,end-7:end-4); 
% Yind = data(:,end-3:end);

% Initialize output array for each threshold value 
% Each cell will end up as 100x1 double 
for t = 1:numel(threshold_vals)
    s_output_rbf{t} = []; 
end 

for t = 1:numel(threshold_vals)
    ad_output_rbf{t} = []; 
end 

for t = 1:numel(threshold_vals)
    s_output_ard{t} = []; 
end 

for t = 1:numel(threshold_vals)
    ad_output_ard{t} = []; 
end 

% Initialize requests array for each threshold value 
% Each cell will end up as 100x1 double 
for t = 1:numel(threshold_vals)
    ad_requests_rbf{t} = []; 
    ad_req_vec_rbf{t} = [];
end 

for t = 1:numel(threshold_vals)
    ad_requests_ard{t} = []; 
    ad_req_vec_ard{t} = [];
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
        
        % Build & train GP model (with & without slices) 
        results = kernels{kern}(X, Y, Yind, tr_ind, tst_ind, inf_func, threshold_vals); 
        
        % For each threshold value: 
        for t = 1:numel(threshold_vals)
            
            disp(['----- Kernel ' kern_id(kern, 1:end) ' | Threshold ' num2str(threshold_vals(t))])
            
            threshold = threshold_vals(t);
            thresh_results = results{t}; 
            % For each test patient: 
            all_s_tst_error = []; 
            all_ad_tst_error = []; 
            all_reqs = []; 
            all_req_vec = [];
            for j = 1:numel(tst_ind)
                % Get corresponding predictions, ground truth, indicators 
                pat_m_s = thresh_results.m_s{j}; 
                pat_m_ad = thresh_results.m_ad{j};
                pat_gt = thresh_results.gt{j}; 
                pat_yind = thresh_results.yind{j}; 
                
                non_zero_inds = find(pat_yind~=0); 
                pat_gt_processed = pat_gt(non_zero_inds); 
                pat_m_s_processed = pat_m_s(non_zero_inds);
                pat_m_ad_processed = pat_m_ad(non_zero_inds); 
                
                source_pat_error = compute_mse(pat_gt_processed, pat_m_s_processed); 
                adapt_pat_error = compute_mse(pat_gt_processed, pat_m_ad_processed); 
                
                all_s_tst_error = [all_s_tst_error; source_pat_error];
                all_ad_tst_error = [all_ad_tst_error; adapt_pat_error]; 
                
                all_reqs = [all_reqs; thresh_results.reqs{j}]; 
                
                all_req_vec = [all_req_vec; thresh_results.req_vec{j}']; 
            end 
            
            if kern == 1 
                s_output_rbf{t} = [s_output_rbf{t}; all_s_tst_error]; 
                ad_output_rbf{t} = [ad_output_rbf{t}; all_ad_tst_error]; 
                ad_requests_rbf{t} = [ad_requests_rbf{t}; all_reqs]; 
                ad_req_vec_rbf{t} = [ad_req_vec_rbf{t}; all_req_vec]; 
            else 
                s_output_ard{t} = [s_output_ard{t}; all_s_tst_error]; 
                ad_output_ard{t} = [ad_output_ard{t}; all_ad_tst_error]; 
                ad_requests_ard{t} = [ad_requests_ard{t}; all_reqs]; 
                ad_req_vec_ard{t} = [ad_req_vec_ard{t}; all_req_vec]; 
            end 
            
        end 
    end 
end 

% Output results, errors, and requests to CSV 
% Get group indices 
group_class = csvread([currentDir '/patient_classification.csv']); 
one_ind = find(group_class(1:end, 2) == 1); 
two_ind = find(group_class(1:end, 2) == 2); 
three_ind = find(group_class(1:end, 2) == 3); 

% s_output_rbf
% For each threshold value: 
for t = 1:numel(threshold_vals)
    
    out = [rid s_output_rbf{t}]; 
    csvwrite([currentDir '/ml_exp_results/t' num2str(threshold_vals(t)) '_sGP_rbf.csv'], out); 
    
    g1 = out(one_ind, 1:end); 
    g2 = out(two_ind, 1:end); 
    g3 = out(three_ind, 1:end); 
    
    csvwrite([currentDir '/ml_exp_results/t' num2str(threshold_vals(t)) '_sGP_rbf_g1.csv'], g1); 
    csvwrite([currentDir '/ml_exp_results/t' num2str(threshold_vals(t)) '_sGP_rbf_g2.csv'], g2); 
    csvwrite([currentDir '/ml_exp_results/t' num2str(threshold_vals(t)) '_sGP_rbf_g3.csv'], g3); 
    
    group_errors = [mean(g1(1:end, 2)) mean(g2(1:end, 2)) mean(g3(1:end, 2))]; 
    csvwrite([currentDir '/ml_exp_results/t' num2str(threshold_vals(t)) '_sGP_rbf_gError.csv'], group_errors); 
    
end 

% ad_output_rbf
for t = 1:numel(threshold_vals)
    
    out = [rid ad_output_rbf{t}]; 
    csvwrite([currentDir '/ml_exp_results/t' num2str(threshold_vals(t)) '_pGP_rbf.csv'], out); 
    
    g1 = out(one_ind, 1:end); 
    g2 = out(two_ind, 1:end); 
    g3 = out(three_ind, 1:end); 
    
    csvwrite([currentDir '/ml_exp_results/t' num2str(threshold_vals(t)) '_pGP_rbf_g1.csv'], g1); 
    csvwrite([currentDir '/ml_exp_results/t' num2str(threshold_vals(t)) '_pGP_rbf_g2.csv'], g2); 
    csvwrite([currentDir '/ml_exp_results/t' num2str(threshold_vals(t)) '_pGP_rbf_g3.csv'], g3); 
    
    group_errors = [mean(g1(1:end, 2)) mean(g2(1:end, 2)) mean(g3(1:end, 2))]; 
    csvwrite([currentDir '/ml_exp_results/t' num2str(threshold_vals(t)) '_pGP_rbf_gError.csv'], group_errors); 
    
    all_reqs = [rid ad_requests_rbf{t}]; 
    csvwrite([currentDir '/ml_exp_results/t' num2str(threshold_vals(t)) '_pGP_rbf_reqs.csv'], out); 
    
    g1_req = all_reqs(one_ind, 1:end); 
    g2_req = all_reqs(two_ind, 1:end); 
    g3_req = all_reqs(three_ind, 1:end); 
    
    csvwrite([currentDir '/ml_exp_results/t' num2str(threshold_vals(t)) '_pGP_rbf_g1_reqs.csv'], g1_req); 
    csvwrite([currentDir '/ml_exp_results/t' num2str(threshold_vals(t)) '_pGP_rbf_g2_reqs.csv'], g2_req); 
    csvwrite([currentDir '/ml_exp_results/t' num2str(threshold_vals(t)) '_pGP_rbf_g3_reqs.csv'], g3_req); 
    
    group_reqs = [sum(g1_req(1:end, 2)) sum(g2_req(1:end, 2)) sum(g3_req(1:end, 2))]; 
    csvwrite([currentDir '/ml_exp_results/t' num2str(threshold_vals(t)) '_pGP_rbf_gReqs.csv'], group_reqs); 
    
    all_req_vec = [rid ad_req_vec_rbf{t}]; 
    csvwrite([currentDir '/ml_exp_results/t' num2str(threshold_vals(t)) '_pGP_rbf_req_vec.csv'], all_req_vec); 
    
    g1_req_vec = all_req_vec(one_ind, 1:end); 
    g2_req_vec = all_req_vec(two_ind, 1:end); 
    g3_req_vec = all_req_vec(three_ind, 1:end); 
    
    csvwrite([currentDir '/ml_exp_results/t' num2str(threshold_vals(t)) '_pGP_rbf_g1_req_vec.csv'], g1_req_vec); 
    csvwrite([currentDir '/ml_exp_results/t' num2str(threshold_vals(t)) '_pGP_rbf_g2_req_vec.csv'], g2_req_vec); 
    csvwrite([currentDir '/ml_exp_results/t' num2str(threshold_vals(t)) '_pGP_rbf_g3_req_vec.csv'], g3_req_vec); 
end 

% s_output_ard
for t = 1:numel(threshold_vals)
    
    out = [rid s_output_ard{t}]; 
    csvwrite([currentDir '/ml_exp_results/t' num2str(threshold_vals(t)) '_sGP_ard.csv'], out); 
    
    g1 = out(one_ind, 1:end); 
    g2 = out(two_ind, 1:end); 
    g3 = out(three_ind, 1:end); 
    
    csvwrite([currentDir '/ml_exp_results/t' num2str(threshold_vals(t)) '_sGP_ard_g1.csv'], g1); 
    csvwrite([currentDir '/ml_exp_results/t' num2str(threshold_vals(t)) '_sGP_ard_g2.csv'], g2); 
    csvwrite([currentDir '/ml_exp_results/t' num2str(threshold_vals(t)) '_sGP_ard_g3.csv'], g3); 
    
    group_errors = [mean(g1(1:end, 2)) mean(g2(1:end, 2)) mean(g3(1:end, 2))]; 
    csvwrite([currentDir '/ml_exp_results/t' num2str(threshold_vals(t)) '_sGP_ard_gError.csv'], group_errors); 
    
end 

% ad_output_ard
for t = 1:numel(threshold_vals)
    
    out = [rid ad_output_ard{t}]; 
    csvwrite([currentDir '/ml_exp_results/t' num2str(threshold_vals(t)) '_pGP_ard.csv'], out); 
    
    g1 = out(one_ind, 1:end); 
    g2 = out(two_ind, 1:end); 
    g3 = out(three_ind, 1:end); 
    
    csvwrite([currentDir '/ml_exp_results/t' num2str(threshold_vals(t)) '_pGP_ard_g1.csv'], g1); 
    csvwrite([currentDir '/ml_exp_results/t' num2str(threshold_vals(t)) '_pGP_ard_g2.csv'], g2); 
    csvwrite([currentDir '/ml_exp_results/t' num2str(threshold_vals(t)) '_pGP_ardf_g3.csv'], g3); 
    
    group_errors = [mean(g1(1:end, 2)) mean(g2(1:end, 2)) mean(g3(1:end, 2))]; 
    csvwrite([currentDir '/ml_exp_results/t' num2str(threshold_vals(t)) '_pGP_ard_gError.csv'], group_errors); 
    
    all_reqs = [rid ad_requests_ard{t}]; 
    csvwrite([currentDir '/ml_exp_results/t' num2str(threshold_vals(t)) '_pGP_ard_reqs.csv'], out); 
    
    g1_req = all_reqs(one_ind, 1:end); 
    g2_req = all_reqs(two_ind, 1:end); 
    g3_req = all_reqs(three_ind, 1:end); 
    
    csvwrite([currentDir '/ml_exp_results/t' num2str(threshold_vals(t)) '_pGP_ard_g1_reqs.csv'], g1_req); 
    csvwrite([currentDir '/ml_exp_results/t' num2str(threshold_vals(t)) '_pGP_ard_g2_reqs.csv'], g2_req); 
    csvwrite([currentDir '/ml_exp_results/t' num2str(threshold_vals(t)) '_pGP_ard_g3_reqs.csv'], g3_req); 
    
    group_reqs = [sum(g1_req(1:end, 2)) sum(g2_req(1:end, 2)) sum(g3_req(1:end, 2))]; 
    csvwrite([currentDir '/ml_exp_results/t' num2str(threshold_vals(t)) '_pGP_ard_gReqs.csv'], group_reqs); 
    
    all_req_vec = [rid ad_req_vec_ard{t}]; 
    csvwrite([currentDir '/ml_exp_results/t' num2str(threshold_vals(t)) '_pGP_ard_req_vec.csv'], all_req_vec); 
    
    g1_req_vec = all_req_vec(one_ind, 1:end); 
    g2_req_vec = all_req_vec(two_ind, 1:end); 
    g3_req_vec = all_req_vec(three_ind, 1:end); 
    
    csvwrite([currentDir '/ml_exp_results/t' num2str(threshold_vals(t)) '_pGP_ard_g1_req_vec.csv'], g1_req_vec); 
    csvwrite([currentDir '/ml_exp_results/t' num2str(threshold_vals(t)) '_pGP_ard_g2_req_vec.csv'], g2_req_vec); 
    csvwrite([currentDir '/ml_exp_results/t' num2str(threshold_vals(t)) '_pGP_ard_g3_req_vec.csv'], g3_req_vec); 
    
end 
