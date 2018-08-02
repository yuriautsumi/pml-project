function [f, df] = gpoe_train(hyp, inf, meanfunc, covfunc, likfunc, x, y, tr_ind_source)

if isfield(hyp, 'w')
    df = struct('cov', 0*hyp.cov, 'lik', 0*hyp.lik, 'w', 0*hyp.w);
else
    df = struct('cov', 0*hyp.cov, 'lik', 0*hyp.lik);
end

y_s0 = y(cat(1,tr_ind_source{:}),:);
y_s0(y_s0==-1) = 0;
all_occ = sum(y_s0);


f = 0;
num_source_subj = numel(tr_ind_source);
for i = 1:num_source_subj
%     x_s = standardize(x(tr_ind_source{i},:), '0');
    x_s = x(tr_ind_source{i},:);
    y_s = y(tr_ind_source{i},:);
    
    if isfield(hyp, 'w')
        y_s0 = y_s; y_s0(y_s0==-1) = 0;
        occ_w = 1;%sum(y_s0) ./ all_occ + eps;
        w2_orig = occ_w'.*exp(2*hyp.w);
        sum_w = sum(w2_orig);
        w2 = w2_orig./sum_w;
        dw = (sum_w - w2_orig) / sum_w^2;
        hyp.w = log(1./sqrt(w2));
    end
    
    [tmp_nlZ, tmp_dnlZ] = gp(hyp, inf, meanfunc, covfunc, likfunc, x_s, y_s);
    f = f + tmp_nlZ;
    df.cov = df.cov + tmp_dnlZ.cov;
    df.lik = df.lik + tmp_dnlZ.lik;
    if isfield(hyp, 'w')
        tmp_dnlZ.w = -tmp_dnlZ.w ./ w2;
        tmp_dnlZ.w = w2_orig.*((w2/sum_w + dw).*tmp_dnlZ.w - w2'*tmp_dnlZ.w/sum_w);
        df.w = df.w + tmp_dnlZ.w;
    end
end

% p = 10;
% max_snr = 10;
% max_ls = 30;
% 
% ll = hyp.cov(1);
% if isfield(hyp, 'w')
%     lw2 = hyp.w;
% else
%     lw2 = 0;
% end
% lsf2 = hyp.cov(2);
% lsn2 = hyp.lik;
% 
% f = f + sum((ll./log(max_ls)).^p);   % length-scales
% df.cov(1) = df.cov(1) + p*(ll).^(p-1)/log(max_ls)^p;
% 
% 
% f = f + sum(((lsf2 - lw2 - lsn2)/log(max_snr)).^p);
% if isfield(hyp, 'w')
%     df.w = df.w - p*(lsf2 - lw2 - lsn2).^(p-1)/log(max_snr)^p;
%     df.cov(2) = df.cov(2) + p*sum((lsf2 - lw2 - lsn2).^(p-1)/log(max_snr)^p);
%     df.lik = df.lik - p*sum((lsf2 - lw2 - lsn2).^(p-1)/log(max_snr)^p);
%     
%     df.w = df.w ./ w2;
%     df.w = w2_orig.*((w2/sum_w + dw).*df.w - w2'*df.w/sum_w);
% else
%     df.cov(2) = df.cov(2) + p*(lsf2 - lsn2).^(p-1)/log(max_snr)^p;
%     df.lik = df.lik - p*sum((lsf2 - lsn2).^(p-1)/log(max_snr)^p);
% end


end