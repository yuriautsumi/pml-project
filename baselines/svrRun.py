## Runs SVR Baseline Model on ADNI Dataset 

from baselineHelperFxn import * 
from sklearn.svm import SVR 
from sklearn.multioutput import MultiOutputRegressor 

import os
import numpy as np 

if __name__ == '__main__':
#    ## Test run 
#    X = np.sort(5 * np.random.rand(40, 1), axis=0)
#    y1 = np.sin(X).ravel()
#    y1[::5] += 3 * (0.5 - np.random.rand(8))
#    y2 = np.sin(X).ravel()
#    y2[::5] += 3 * (0.5 - np.random.rand(8))
#    y3 = np.sin(X).ravel()
#    y3[::5] += 3 * (0.5 - np.random.rand(8))
#    y1, y2, y3 = np.reshape(y1, (40,1)), np.reshape(y2, (40,1)), np.reshape(y3, (40,1))
#    y = np.hstack((y1,y2,y3))
#    
#    X_tr, Y_tr, X_te, Y_te = splitArrays(X, y) 
#    
#    multiSVR = MultiOutputRegressor(SVR(kernel='rbf'))
#    multiSVR.fit(X_tr, Y_tr)
#    Y_hat = multiSVR.predict(X_te)
#    
#    print('Results:', np.mean(abs(Y_hat-Y_te)))
    
    ## ADNI Data 
    # Define directories
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(CURRENT_DIR, 'adni_adas13_100_fl1_l4.csv')
    
    data = np.genfromtxt(DATA_DIR, delimiter=',')
    X_all = data[:, 3:-8]
    Y_all = data[:, -8:-4]
    ind_all = data[:, -4:-3]
    
    X_tr, Y_tr, X_te, Y_te = splitArrays(X_all, Y_all) 
    
    multiSVR = MultiOutputRegressor(SVR(kernel='rbf'), n_jobs=4)
    multiSVR.fit(X_tr, Y_tr)
    Y_hat = multiSVR.predict(X_te)
    
    print('Results:', np.mean(abs(Y_hat-Y_te)))