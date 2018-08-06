## Runs Linear Regression Baseline Model on ADNI Dataset 

from baselineHelperFxn import * 
from sklearn.linear_model import LinearRegression 

import os
import pathlib
import numpy as np 

if __name__ == '__main__':
    # Define directories
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(CURRENT_DIR, 'adni_adas13_100_fl1_l4.csv')
    RESULTS_FOLDER_DIR = os.path.join(CURRENT_DIR, 'Results')
    pathlib.Path(RESULTS_FOLDER_DIR).mkdir(parents=True, exist_ok=True) 
    FINAL_RESULTS_DIR = os.path.join(RESULTS_FOLDER_DIR, 'linreg_results.csv')
    FINAL_ERROR_DIR = os.path.join(RESULTS_FOLDER_DIR, 'linreg_errors.csv')
    FINAL_ERROR_SUMMARY_DIR = os.path.join(RESULTS_FOLDER_DIR, 'linreg_errors_summary.csv')

    data = np.genfromtxt(DATA_DIR, delimiter=',')
    ID_all = data[:, 0:1]
    X_all = data[:, 3:-8]
    Y_all = data[:, -8:-4]
    ind_all = data[:, -4:]

    results = None 
    # Loop over folds 
    for f in range(10): 

        te_list = list(range(f*10, f*10+10))
        tr_list = list(set(list(range(100))) - set(te_list))
    
        X_tr, Y_tr, ind_tr, ID_tr, X_te, Y_te, ind_te, ID_te = splitArrays(ID_all, X=X_all, Y=Y_all, ind=ind_all, tr_split=tr_list) 
        
        reg = LinearRegression() 
        reg.fit(X_tr, Y_tr) 
        Y_hat = reg.predict(X_te) 

        results = np.hstack((ID_te, Y_hat, Y_te, ind_te)) if results is None else np.vstack((results, np.hstack((ID_te, Y_hat, Y_te, ind_te))))

    np.savetxt(FINAL_RESULTS_DIR, results, delimiter=',')
    process_results(FINAL_RESULTS_DIR, FINAL_ERROR_DIR, FINAL_ERROR_SUMMARY_DIR)