## Processes ADNI Dataset 
# 80/10/10 split 

# Import libraries 
import os
import pathlib
import numpy as np 
import pandas as pd

# Define directories
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, 'adni_adas13_100_fl1_l4.csv')
LOS_FOLDER_DIR = os.path.join(CURRENT_DIR, 'data/length-of-stay')
pathlib.Path(LOS_FOLDER_DIR).mkdir(parents=True, exist_ok=True)

data = np.genfromtxt(DATA_DIR, delimiter=',')
X_all = np.array(data[:, 3:-8], dtype='object')
Y_all = data[:, -8:-7]
ind_all = data[:, -4:-3]
ID = np.unique(data[:,:1]).tolist()
ID_all = data[:,0]

# TODO: 
# Replace 0's with blank cells 
X_all[np.where(X_all==0)] = '' 

#/data/length-of-stay/foldX/test 
#/data/length-of-stay/foldX/train 
#/data/length-of-stay/XX_listfile.csv 

# For each fold: 
for f in range(10): 
    fold = f+1 
    
    testID = ID[f*10:f*10+10] 
    valID = ID[((f+1)%10)*10:((f+1)%10)*10+10] 
    trainID = np.setdiff1d(np.setdiff1d(ID, testID), valID).tolist() 
    
    # Make folders 
    FOLD_FOLDER_DIR = os.path.join(LOS_FOLDER_DIR, 'f%s'%(fold)) 
    TEST_FOLDER_DIR = os.path.join(FOLD_FOLDER_DIR, 'test') 
    TRAIN_FOLDER_DIR = os.path.join(FOLD_FOLDER_DIR, 'train') 
    
    pathlib.Path(FOLD_FOLDER_DIR).mkdir(parents=True, exist_ok=True) 
    pathlib.Path(TEST_FOLDER_DIR).mkdir(parents=True, exist_ok=True) 
    pathlib.Path(TRAIN_FOLDER_DIR).mkdir(parents=True, exist_ok=True) 
    
    # Notation: 
        # ID#_episode#_timeseries.csv 
        # train_listfile.csv 
    
    ## TRAINING 
    
    lf_array = np.array([['stay', 'period_length', 'y_true']], dtype='object') 
    
    # For each train patient: 
    for tr in trainID: 
        # Grab appropriate rows 
        row_inds = np.where(ID_all == tr)[0] 
        
        X = X_all[row_inds] 
        Y = Y_all[row_inds] 
        ind = ind_all[row_inds] 
        ts_header = np.array([['a']*X.shape[1]], dtype='object') 
        
        real_inds = np.where(ind == 1)[0] 
        
        tr_X_all = np.array(X[real_inds], dtype='object') 
        tr_Y_all = Y[real_inds] 
        
        # Construct & save timeseries data 
        TS_DIR = os.path.join(TRAIN_FOLDER_DIR, '%s_episode1_timeseries.csv'%(int(tr))) 
        ts_array = np.vstack((ts_header, tr_X_all))
#        np.savetxt(TS_DIR, ts_array, delimiter=',', fmt='object') 
        df = pd.DataFrame(ts_array)
        df.to_csv(TS_DIR, index=False, header=False)
        
        # Construct lisfile data 
        for r in real_inds: 
            lf_row = np.array([['%s_episode1_timeseries.csv'%(int(tr)), r+1., float(Y[r])]], dtype='object') 
            lf_array = np.vstack((lf_array, lf_row)) 
    
    # Save listfile data 
    LF_DIR = os.path.join(FOLD_FOLDER_DIR, 'train_listfile.csv')
#    np.savetxt(LF_DIR, lf_array, delimiter=',')
    df = pd.DataFrame(lf_array)
    df.to_csv(LF_DIR, index=False, header=False)
    
    ## TESTING 
    
    lf_array = np.array([['stay', 'period_length', 'y_true']], dtype='object')
    
    # For each test patient: 
    for te in testID: 
        # Grab appropriate rows 
        row_inds = np.where(ID_all == te)[0] 
        
        X = X_all[row_inds] 
        Y = Y_all[row_inds] 
        ind = ind_all[row_inds] 
        ts_header = np.array([['a']*X.shape[1]], dtype='object')
        
        real_inds = np.where(ind == 1)[0] 
        
        te_X_all = np.array(X[real_inds], dtype='object')
        te_Y_all = Y[real_inds]  
        
        # Construct & save timeseries data 
        TS_DIR = os.path.join(TEST_FOLDER_DIR, '%s_episode1_timeseries.csv'%(int(te))) 
        ts_array = np.vstack((ts_header, te_X_all))
#        np.savetxt(TS_DIR, ts_array, delimiter=',', fmt='object') 
        df = pd.DataFrame(ts_array)
        df.to_csv(TS_DIR, index=False, header=False)
        
        # Construct lisfile data 
        for r in real_inds: 
            lf_row = np.array([['%s_episode1_timeseries.csv'%(int(te)), r+1., float(Y[r])]], dtype='object') 
            lf_array = np.vstack((lf_array, lf_row)) 
    
    # Save listfile data 
    LF_DIR = os.path.join(FOLD_FOLDER_DIR, 'test_listfile.csv')
#    np.savetxt(LF_DIR, lf_array, delimiter=',')
    df = pd.DataFrame(lf_array)
    df.to_csv(LF_DIR, index=False, header=False)
    
    
    ## VALIDATION 
    
    lf_array = np.array([['stay', 'period_length', 'y_true']], dtype='object')
    
    # For each validation patient: 
    for val in valID: 
        # Grab appropriate rows 
        row_inds = np.where(ID_all == val)[0] 
        
        X = X_all[row_inds] 
        Y = Y_all[row_inds] 
        ind = ind_all[row_inds] 
        ts_header = np.array([['a']*X.shape[1]], dtype='object')
        
        real_inds = np.where(ind == 1)[0] 
        
        val_X_all = np.array(X[real_inds], dtype='object')
        val_Y_all = Y[real_inds] 
        
        # Construct listfile data 
        for r in real_inds: 
            lf_row = np.array([['%s_episode1_timeseries.csv'%(int(val)), r+1., float(Y[r])]], dtype='object') 
            lf_array = np.vstack((lf_array, lf_row)) 
    
    # Save listfile data 
    LF_DIR = os.path.join(FOLD_FOLDER_DIR, 'val_listfile.csv')
#    np.savetxt(LF_DIR, lf_array, delimiter=',')
    df = pd.DataFrame(lf_array)
    df.to_csv(LF_DIR, index=False, header=False)
