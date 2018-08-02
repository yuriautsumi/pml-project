function [updatable, updatable_visit] = check_pgp_update(req_visits, ad_visits, pat_pgp_m) 
% Summmary: Checks if pGP can be updated. 
% Input: list of requested visits, list of adaptation data visits, accumulated predictions 
% Output: updatable visit, boolean (updatable or not) 

% Find visits eligible as adaptation data 
if isempty(find(isnan(pat_pgp_m),1))
    first_nan = 0; 
else 
    first_nan = find(isnan(pat_pgp_m),1); % first occurrence of NaN 
end 
eligible_visit_max = max(0, first_nan-1-4); 
eligible_visits = linspace(1, eligible_visit_max, eligible_visit_max); 

% Find visits updatable as adaptation data 
available_visits = intersect(eligible_visits, req_visits); 
updatable_visit = available_visits(~ismember(available_visits, ad_visits)); 

updatable = ~isempty(updatable_visit); 
end         