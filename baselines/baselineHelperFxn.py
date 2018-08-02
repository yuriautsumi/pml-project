## Helper functions for Baselines

import numpy as np

def splitArrays(X=None, Y=None, tr_split=0.9, **kwargs): 
    '''
    Splits arrays into training and testing sets 
    '''
    # Split X 
    X_tr, X_te = None, None 
    if X is not(None):
        N, _ = X.shape
        cutoff = int(N*tr_split)
        X_tr = X[:cutoff]
        X_te = X[cutoff:]
    
    # Split Y 
    Y_tr, Y_te = None, None 
    if Y is not(None):
        N, _ = Y.shape
        cutoff = int(N*tr_split)
        Y_tr = Y[:cutoff]
        Y_te = Y[cutoff:]
    
    return X_tr, Y_tr, X_te, Y_te 

def reshapeArrays(window_size, X=None, Y=None, **kwargs):
    ''' 
    Reshapes input arrays to be compatible with LSTMs 
    ''' 
    # Reshape X 
    if X is not(None): 
        N, num_features = X.shape
        X_reshaped = np.zeros((N-window_size, window_size, num_features))
        
        for i in range(N-window_size):
            current_window = X[i:i+window_size, :]
            X_reshaped[i, :, :] = current_window
    
    # Reshape Y 
    if Y is not(None): 
        Y_reshaped = Y[window_size:]
        
    return X_reshaped, Y_reshaped 

if __name__ == '__main__':
    pass 