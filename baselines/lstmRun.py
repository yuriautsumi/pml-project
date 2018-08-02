## Runs LSTM Baseline Model on ADNI Dataset 

from baselineHelperFxn import * 
from lstmBaseline import * 

import os
import numpy as np 

if __name__ == '__main__':
    
#    ## Test run 
#    X = np.random.rand(300, 5)
#    Y = np.random.rand(300, 3)
#    
#    X_tr, Y_tr, X_te, Y_te = splitArrays(X, Y) 
#    
#    window_size = 5 
#    
#    X_tr, Y_tr = reshapeArrays(window_size, X_tr, Y_tr)
#    X_te, Y_te = reshapeArrays(window_size, X_te, Y_te)
#    
#    myNet = Network(window_size, 15, 5, 3, 1)
#    myNet.trainModel(X_tr, Y_tr)
#    Y_hat = myNet.predictModel(X_te) 
    
    ## ADNI Data 
    # Define directories
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(CURRENT_DIR, 'adni_adas13_100_fl1_l4.csv')
    
    data = np.genfromtxt(DATA_DIR, delimiter=',')
    X_all = data[:, 3:-8]
    Y_all = data[:, -8:-4]
    ind_all = data[:, -4:-3]
    
    X_tr_original, Y_tr_original, X_te_original, Y_te_original = splitArrays(X_all, Y_all) 
    
    window_list = [2,3,4]
    results = np.zeros((len(window_list), 2))
    
    for w in window_list: 
        window_size = w 
        depth = 1
        
        X_tr, Y_tr = reshapeArrays(window_size, X_tr_original, Y_tr_original)
        X_te, Y_te = reshapeArrays(window_size, X_te_original, Y_te_original)
        
        myNet = Network(window_size, 2100*0.1-window_size, X_all.shape[1], Y_all.shape[1], depth)
        myNet.trainModel(X_tr, Y_tr, epochs=15)
        Y_hat = myNet.predictModel(X_te) 
        
        print(np.mean(abs(Y_hat-Y_te)))
        
        results[window_list.index(w), :] = [w, np.mean(abs(Y_hat-Y_te))]
    
    print('Results:', results)