function save_results(t, threshold_vals, error_array, output_array, req_array, req_vec, ad_vec, rid, one_ind, two_ind, three_ind, currentDir, sgp_bool, rbf_bool)

num_patients = 100; 

model_ext = ['mGP'; 'sGP']; 
kernel_ext = ['ard'; 'rbf']; 

model = model_ext(sgp_bool+1, 1:end); 
kernel = kernel_ext(rbf_bool+1, 1:end); 

% Save outputs 
out_ids = repmat(rid', [length(output_array{t})/num_patients 1]); 
output = [out_ids(:) output_array{t}]; 
csvwrite([currentDir '/ml_4ts_exp_results/t' num2str(threshold_vals(t)) '_' model '_' kernel '_OUTPUT.csv'], output); 

% g1o = output(one_ind, 1:end); 
% g2o = output(two_ind, 1:end); 
% g3o = output(three_ind, 1:end); 
% 
% csvwrite([currentDir '/ml_4ts_exp_results/t' num2str(threshold_vals(t)) '_' model '_' kernel '_OUTPUT_g1.csv'], g1o); 
% csvwrite([currentDir '/ml_4ts_exp_results/t' num2str(threshold_vals(t)) '_' model '_' kernel '_OUTPUT_g2.csv'], g2o); 
% csvwrite([currentDir '/ml_4ts_exp_results/t' num2str(threshold_vals(t)) '_' model '_' kernel '_OUTPUT_g3.csv'], g3o); 

% Save errors 
errors = [rid error_array{t}];
csvwrite([currentDir '/ml_4ts_exp_results/t' num2str(threshold_vals(t)) '_' model '_' kernel '_ERROR.csv'], errors); 
    
g1 = errors(one_ind, 1:end); 
g2 = errors(two_ind, 1:end); 
g3 = errors(three_ind, 1:end); 

csvwrite([currentDir '/ml_4ts_exp_results/t' num2str(threshold_vals(t)) '_' model '_' kernel '_ERROR_g1.csv'], g1); 
csvwrite([currentDir '/ml_4ts_exp_results/t' num2str(threshold_vals(t)) '_' model '_' kernel '_ERROR_g2.csv'], g2); 
csvwrite([currentDir '/ml_4ts_exp_results/t' num2str(threshold_vals(t)) '_' model '_' kernel '_ERROR_g3.csv'], g3); 

group_errors = [mean(g1(1:end, 2)) mean(g2(1:end, 2)) mean(g3(1:end, 2))]; 
csvwrite([currentDir '/ml_4ts_exp_results/t' num2str(threshold_vals(t)) '_' model '_' kernel '_gERROR.csv'], group_errors); 

% Save the following for meta-learning GPs 
if ~sgp_bool
    % Save requests 
    all_reqs = [rid req_array{t}]; 
    csvwrite([currentDir '/ml_4ts_exp_results/t' num2str(threshold_vals(t)) '_' model '_' kernel '_reqs.csv'], all_reqs); 

    g1_req = all_reqs(one_ind, 1:end); 
    g2_req = all_reqs(two_ind, 1:end); 
    g3_req = all_reqs(three_ind, 1:end); 

    csvwrite([currentDir '/ml_4ts_exp_results/t' num2str(threshold_vals(t)) '_' model '_' kernel '_g1_reqs.csv'], g1_req); 
    csvwrite([currentDir '/ml_4ts_exp_results/t' num2str(threshold_vals(t)) '_' model '_' kernel '_g2_reqs.csv'], g2_req); 
    csvwrite([currentDir '/ml_4ts_exp_results/t' num2str(threshold_vals(t)) '_' model '_' kernel '_g3_reqs.csv'], g3_req); 

    group_reqs = [sum(g1_req(1:end, 2)) sum(g2_req(1:end, 2)) sum(g3_req(1:end, 2))]; 
    csvwrite([currentDir '/ml_4ts_exp_results/t' num2str(threshold_vals(t)) '_' model '_' kernel '_gReqs.csv'], group_reqs); 
    
    % Save requests vector 
    all_req_vec = [rid req_vec{t}]; 
    csvwrite([currentDir '/ml_4ts_exp_results/t' num2str(threshold_vals(t)) '_' model '_' kernel '_req_vec.csv'], all_req_vec); 

    g1_req_vec = all_req_vec(one_ind, 1:end); 
    g2_req_vec = all_req_vec(two_ind, 1:end); 
    g3_req_vec = all_req_vec(three_ind, 1:end); 

    csvwrite([currentDir '/ml_4ts_exp_results/t' num2str(threshold_vals(t)) '_' model '_' kernel '_g1_req_vec.csv'], g1_req_vec); 
    csvwrite([currentDir '/ml_4ts_exp_results/t' num2str(threshold_vals(t)) '_' model '_' kernel '_g2_req_vec.csv'], g2_req_vec); 
    csvwrite([currentDir '/ml_4ts_exp_results/t' num2str(threshold_vals(t)) '_' model '_' kernel '_g3_req_vec.csv'], g3_req_vec); 
    
    % Save adaptation vector 
    all_ad_vec = [rid ad_vec{t}]; 
    csvwrite([currentDir '/ml_4ts_exp_results/t' num2str(threshold_vals(t)) '_' model '_' kernel '_ad_vec.csv'], all_ad_vec); 
    
    g1_ad_vec = all_ad_vec(one_ind, 1:end); 
    g2_ad_vec = all_ad_vec(two_ind, 1:end); 
    g3_ad_vec = all_ad_vec(three_ind, 1:end); 

    csvwrite([currentDir '/ml_4ts_exp_results/t' num2str(threshold_vals(t)) '_' model '_' kernel '_g1_ad_vec.csv'], g1_ad_vec); 
    csvwrite([currentDir '/ml_4ts_exp_results/t' num2str(threshold_vals(t)) '_' model '_' kernel '_g2_ad_vec.csv'], g2_ad_vec); 
    csvwrite([currentDir '/ml_4ts_exp_results/t' num2str(threshold_vals(t)) '_' model '_' kernel '_g3_ad_vec.csv'], g3_ad_vec); 
end 
end  