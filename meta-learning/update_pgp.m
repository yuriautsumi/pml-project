function [x_ad, y_ad, ad_visits] = update_pgp(updatable, updatable_visit, ad_visits, xy_req, pat_pgp_m, x_ad, y_ad)
% Summmary: Updates pGP as necessary. 
% Input: updatable boolean, updatable visit, visits used for adaptation, 
% requested data, predictions, x_ad, y_ad 
% Output: x_ad, y_ad 

if updatable 
    for uv = updatable_visit'
%         disp('visit')
%         disp(uv)
%         disp('xad shape')
%         disp(size(x_ad))
%         disp('xy req shape')
%         disp(size(xy_req{uv}))
%         disp('yad shape')
%         disp(size(y_ad))
%         disp('y req shape')
%         disp(size(pat_pgp_m(uv+1:uv+4)'))
        x_ad = [x_ad; xy_req{uv}];
        y_ad = [y_ad; pat_pgp_m(uv+1:uv+4)'];

        ad_visits = [ad_visits; uv]; 
    end 
end 
end 