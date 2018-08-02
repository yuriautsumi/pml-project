function K = covSEard_mod2(hyp, x, z, i)

% Squared Exponential covariance function with Automatic Relevance Detemination
% (ARD) distance measure. The covariance function is parameterized as:
%
% k(x^p,x^q) = sf^2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2)
%
% where the P matrix is diagonal with ARD parameters ell_1^2,...,ell_D^2, where
% D is the dimension of the input space and sf2 is the signal variance. The
% hyperparameters are:
%
% hyp = [ log(ell_1)
%         log(ell_2)
%          .
%         log(ell_D)
%         log(sf) ]
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-09-10.
%
% See also COVFUNCTIONS.M.

% oggi: we need to specify here the number of parameters, which corresponds
% to the number of slices + one more parameter for \sigma (kernel variance)

slices = [2 11 14 243 609 617 618];

if nargin<2, K = '8'; return; end              % report number of parameters  %'(D+1)';
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = isempty(z); dg = strcmp(z,'diag');                       % determine mode

[n,D] = size(x);


% oggi: I built this example assuming only two slices - this has to be defined here:
% it assumes that the first slice is 1:D-1 and second D, where D is the
% number of features in x
%slices = [D-1 D];


%oggi: here I re-define the ell (length scales previosly being separate for
%each input dimension

ell = exp(hyp(1)*ones(1,slices(1)));    

if numel(slices)>1
    for k=2:numel(slices)
        ell = [ell exp(hyp(k)*ones(1,slices(k)-slices(k-1)))];                               % characteristic length scale
    end
end
sf2 = exp(2*hyp(numel(slices)+1));                                         % signal variance

% precompute squared distances
if dg                                                               % vector kxx
  K = zeros(size(x,1),1);
else
    
  if xeqz                                                 % symmetric matrix Kxx
    K = sq_dist(diag(1./ell)*x');
  else                                                   % cross covariances Kxz
    K = sq_dist(diag(1./ell)*x',diag(1./ell)*z');
  end
end

K = sf2*exp(-K/2);                                                  % covariance
if nargin>3                                                        % derivatives
  if i<=numel(slices)%D                                              % length scale parameters
    if dg
      K = K*0;
    else
      %oggi: here we select the slices of x and compute gradients for
      %\lambda (length scale) for each slice
      if xeqz
        if i==1
           K = K.*sq_dist(x(:,1:slices(1))'/exp(hyp(i)));
        else
           K = K.*sq_dist(x(:,slices(i-1)+1:slices(i))'/exp(hyp(i))); 
        end
      else
        if i==1  
           K = K.*sq_dist(x(:,1:slices(1))'/exp(hyp(i)),z(:,1:slices(1))'/exp(hyp(i)));
        else
           K = K.*sq_dist(x(:,slices(i-1)+1:slices(i))'/exp(hyp(i)),z(:,slices(i-1)+1:slices(i))'/exp(hyp(i)));
        end
      end
    end
  elseif i==numel(slices)+1%D+1                                            % magnitude parameter
    K = 2*K;
  else
    error('Unknown hyperparameter')
  end
end