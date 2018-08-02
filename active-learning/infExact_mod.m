function [post, nlZ, dnlZ] = infExact_mod(hyp, mean, cov, lik, x, y, gen)

% Modified version of the original infExact.m function of Rasmussen. It has
% been changed to account for multi-dimensional output y (shared covariance
% function among the dimensions). It can also compute derivatives w.r.t the
% input space x. If the flag 'gen' is activated it acts as a GPLVM and
% calculates the derivatives of x, otherwise it acts as the classical GPR.

if nargin < 7, gen = 0; end
if iscell(lik), likstr = lik{1}; else likstr = lik; end
if ~ischar(likstr), likstr = func2str(likstr); end
if ~strcmp(likstr,'likGauss')               % NOTE: no explicit call to likGauss
  error('Exact inference only possible with Gaussian likelihood');
end

if iscell(x)
    [n, ~] = size(x{1});
else
    [n, ~] = size(x);
end
dim = size(y, 2);

K = feval(cov{:}, hyp.cov, x);                      % evaluate covariance matrix

sn2 = exp(2*hyp.lik);                               % noise variance of likGauss
% if sn2 < 1e-6
    L = chol(K + sn2*eye(n)); sl =   1;            % Cholesky factor of covariance with noise
% else
%     L = chol(K/sn2+eye(n)); sl = sn2;
% end
alpha = solve_chol(L,y)/sl;

post.alpha = alpha;                            % return the posterior parameters
post.L  = L*sqrt(sl);
post.invL = L \ (sqrt(sl)*eye(n));

if nargout>1                               % do we want the marginal likelihood?

    nlZ = trace(alpha*y')/2 + dim*sum(log(diag(L))) + n*dim*log(2*pi*sl)/2;
    if nargout>2                                         % do we want derivatives?
        dnlZ = hyp;                                 % allocate space for derivatives
        dnlZ.cov = zeros(size(dnlZ.cov));
        dnlZ.lik = zeros(size(dnlZ.lik));
        if gen              % if generative, compute derivatives w.r.t x
            dnlZ.x = zeros(size(x));
            gX = feval(cov{:}, hyp.cov, x, [], 100);
        end
        invK = solve_chol(L,eye(n))/sl;
        post.invK = invK;
        invKYYtinvK = alpha*alpha';
        Q = dim*invK - invKYYtinvK;    % precompute for convenience
        for i = 1:numel(hyp.cov)
            dnlZ.cov(i) = dnlZ.cov(i) + sum(sum(Q.*feval(cov{:}, hyp.cov, x, [], i)))/2;
        end
        
        dnlZ.lik = dnlZ.lik + sn2*trace(Q);
        if gen
            gX = permute(gX, [1 3 2]);
            dnlZ.x = squeeze(sum(bsxfun(@times, gX, Q/2)));
        end
        if gen, dnlZ.x = dnlZ.x(:);end
    end
end
