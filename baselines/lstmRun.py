## Runs LSTM Baseline Model on ADNI Dataset 

from baselineHelperFxn import * 
from lstmBaseline import * 

import os 
import pathlib 
import numpy as np 

if __name__ == '__main__':
    # Define directories
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(CURRENT_DIR, 'adni_adas13_100_fl1_l4.csv')
    RESULTS_FOLDER_DIR = os.path.join(CURRENT_DIR, 'Results')
    pathlib.Path(RESULTS_FOLDER_DIR).mkdir(parents=True, exist_ok=True) 
    
    data = np.genfromtxt(DATA_DIR, delimiter=',')
    ID_all = data[:, 0:1]
    X_all = data[:, 3:-8]
    Y_all = data[:, -8:-4]
    ind_all = data[:, -4:]

    window_list = [1,2,3,4,5,6,7,8,9,10]
    results = [None]*len(window_list)
    # Loop over folds 
    for f in range(10): 
        
        te_list = list(range(f*10, f*10+10))
        tr_list = list(set(list(range(100))) - set(te_list))

        X_tr_original, Y_tr_original, ind_tr, ID_tr, X_te_original, Y_te_original, ind_te, ID_te = splitArrays(ID_all, X=X_all, Y=Y_all, ind=ind_all, tr_split=tr_list, lstm_bool=True) 
        
        # Remove first visit index from ID_te and ind_te: 
        list_ID_te = ID_te.flatten().tolist()
        for i in np.unique(ID_te): 
            first_ind = list_ID_te.index(i)
            ID_te = np.delete(ID_te, [first_ind], axis=0)
            ind_te = np.delete(ind_te, [first_ind], axis=0)

        # Loop over windows 
        for w in window_list: 
            window_size = w 
            depth = 1
            
            X_tr_dict, Y_tr_dict = reshapeArrays(window_size, X_tr_original, Y_tr_original)
            X_te_dict, Y_te_dict = reshapeArrays(window_size, X_te_original, Y_te_original)

            X_tr, Y_tr = np.vstack(X_tr_dict.values()), np.vstack(Y_tr_dict.values())
            X_te, Y_te = np.vstack(X_te_dict.values()), np.vstack(Y_te_dict.values())
            
            myNet = Network(window_size, 2100*0.1-window_size, X_all.shape[1], Y_all.shape[1], depth)
            myNet.trainModel(X_tr, Y_tr, epochs=15)
            Y_hat = myNet.predictModel(X_te) 

            results[window_list.index(w)] = np.hstack((ID_te, Y_hat, Y_te, ind_te)) if results[window_list.index(w)] is None else np.vstack((results[window_list.index(w)], np.hstack((ID_te, Y_hat, Y_te, ind_te))))
    
    for w in window_list: 
        FINAL_RESULTS_DIR = os.path.join(RESULTS_FOLDER_DIR, 'w{}_lstm_results.csv'.format(w))
        FINAL_ERROR_DIR = os.path.join(RESULTS_FOLDER_DIR, 'w{}_lstm_errors.csv'.format(w))
        FINAL_ERROR_SUMMARY_DIR = os.path.join(RESULTS_FOLDER_DIR, 'w{}_lstm_errors_summary.csv'.format(w))

        np.savetxt(FINAL_RESULTS_DIR, results[window_list.index(w)], delimiter=',')
        process_results(FINAL_RESULTS_DIR, FINAL_ERROR_DIR, FINAL_ERROR_SUMMARY_DIR)