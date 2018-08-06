## Helper functions for Baselines

import numpy as np

def splitArrays(ID_all, X=None, Y=None, ind=None, tr_split=list(range(0,90)), lstm_bool=False, **kwargs): 
    """
    Splits arrays into training and testing sets 
    
    PARAMETERS 
    ID_all: array of ID column 
    X: array of features 
    Y: array of labels 
    ind: array of indicators 
    tr_split: list of training patient indices 
    lstm_bool: boolean indicating if LSTM data 
    """
    
    ID_unique = np.sort(np.unique(ID_all))
    
    assert len(tr_split) < len(ID_unique)
    
    tr_ID = ID_unique[tr_split]
    te_ID = np.setdiff1d(ID_unique,tr_ID)

    # Get row indices for training and testing data 
    tr_inds = []
    for i in tr_ID: 
        tr_inds.extend(np.where(ID_all == i)[0].tolist())
        
    te_inds = []
    for i in te_ID: 
        te_inds.extend(np.where(ID_all == i)[0].tolist())
    
    if lstm_bool: 
        X_tr, Y_tr = {}, {} 
        for i in tr_ID: 
            X_tr[i] = X[np.where(ID_all == i)[0].tolist()]
            Y_tr[i] = Y[np.where(ID_all == i)[0].tolist()]
            
        X_te, Y_te = {}, {} 
        for i in te_ID: 
            X_te[i] = X[np.where(ID_all == i)[0].tolist()]
            Y_te[i] = Y[np.where(ID_all == i)[0].tolist()]
        
    else: 
        # Split X 
        X_tr, X_te = None, None 
        if X is not(None):
            X_tr = X[tr_inds]
            X_te = X[te_inds]
        
        # Split Y 
        Y_tr, Y_te = None, None 
        if Y is not(None):
            Y_tr = Y[tr_inds]
            Y_te = Y[te_inds]
    
    # Split ind 
    ind_tr, ind_te = None, None 
    if ind is not(None):
        ind_tr = ind[tr_inds]
        ind_te = ind[te_inds]

    # Split IDs 
    ID_tr = ID_all[tr_inds]
    ID_te = ID_all[te_inds]

    return X_tr, Y_tr, ind_tr, ID_tr, X_te, Y_te, ind_te, ID_te

def reshapeArrays(window_size, X=None, Y=None, **kwargs):
    """ 
    Reshapes input arrays to be compatible with LSTMs 

    PARAMETERS 
    window_size: int of window size 
    X: array of features 
    Y: array of labels 
    """
    
    # Reshape X 
    X_final = {}
    if X is not(None):
        for ID in X.keys():
            current_X = X[ID]
            N, num_features = current_X.shape
            X_reshaped = np.zeros((N-1, window_size, num_features))
            
            first_visit = current_window = current_X[:1, :]
            
            for i in range(window_size-1): # 0, 1, 2, 3 
                current_window = current_X[:i+1, :]
                current_window = np.vstack((np.tile(first_visit, ((window_size-1)-i, 1)), current_window))
                X_reshaped[i, :, :] = current_window 
                
            for j in range(N-window_size): 
                current_window = current_X[j:j+window_size, :] 
                X_reshaped[(window_size-1)+j, :, :] = current_window 
                
            X_final[ID] = X_reshaped
    
    # Reshape Y 
    Y_final = {}
    if Y is not(None): 
        for ID in Y.keys(): 
            current_Y = Y[ID]
            Y_final[ID] = current_Y[1:]
        
    return X_final, Y_final 

def __mae(Y, Y_hat):
	"""
    Computes mean absolute error 

    PARAMETERS 
    Y: array of true labels 
    Y_hat: array of predictions 
	"""

	diff = np.subtract(Y, Y_hat)
	abs_diff = np.fabs(diff)

	return sum(abs_diff)/Y_hat.shape[0]

def process_results(FINAL_RESULTS_DIR, FINAL_ERROR_DIR, FINAL_ERROR_SUMMARY_DIR):
    """
    Processes errors from final results 

    PARAMETERS 
    FINAL_RESULTS_DIR: directory of final results 
    FINAL_ERROR_DIR: directory of final errors 
    FINAL_ERROR_SUMMARY_DIR: directory of final error summary 
    """ 
    
    results = np.genfromtxt(FINAL_RESULTS_DIR, delimiter=',')

    ts = (results.shape[1] - 1)//3

    ID_col = results[:, :1]
    Y_hat = results[:, 1:ts+1]
    Y = results[:, ts+1:2*ts+1]
    ind = results[:, 2*ts+1:]

    ID = np.unique(ID_col)

    error_results = np.hstack((ID[:, np.newaxis], np.zeros((len(ID), ts))))
    error_summary = np.zeros((ts, 2))
    # iterate over time steps 
    for i in range(ts): # 0, 1, 2, 3 
        ts_inds = np.where(ind[:, i:i+1] == 1)[0]

        ts_errors = None 
        # iterate over patients 
        for j in ID: 
            pat_inds = np.where(ID_col == j)[0]
            true_pat_inds = np.intersect1d(ts_inds, pat_inds)
            if true_pat_inds.size != 0: 
                pat_mae = __mae(Y[true_pat_inds, i:i+1], Y_hat[true_pat_inds, i:i+1])
                ts_errors = np.array([pat_mae]) if ts_errors is None else np.vstack((ts_errors, np.array([pat_mae])))
        if ts_errors is not(None): 
            error_results[:, i+1:i+2] = ts_errors 

            error_summary[i, 0] = np.mean(ts_errors)
            error_summary[i, 1] = np.std(ts_errors)

    np.savetxt(FINAL_ERROR_DIR, error_results, delimiter=',')
    np.savetxt(FINAL_ERROR_SUMMARY_DIR, error_summary, delimiter=',')
    
    return None 

if __name__ == '__main__':
    pass 