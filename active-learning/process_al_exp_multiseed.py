# Processing AL Experiment Results - Multiseed 

# Import libraries 
import os
import numpy as np 
import pathlib
import matplotlib.pyplot as plt

plt.rcParams['axes.facecolor'] = 'lightgray'
plt.rcParams['font.family'] = 'serif'

## PREPARE DATA  
print('----- PREPARING DATA -----')

# Define directories
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, 'al_exp_seed_results_v2') 
#RESULTS_DIR = os.path.join(CURRENT_DIR, 'al_exp_seed_results') 

GRAPH_FOLDER_DIR = os.path.join(RESULTS_DIR, 'graphs')
pathlib.Path(GRAPH_FOLDER_DIR).mkdir(parents=True, exist_ok=True) 

folds = [1]
#folds = list(range(1, 3))

x_ticks = [100] + list(range(150,1100,50))

rbf_graph_data = []
# For RBF: 
for f in folds:
    for s in [1, 2, 3, 4, 5]:
        # Load CSV data for each model 
        m1_data = np.genfromtxt(os.path.join(RESULTS_DIR, 'f%s_s%s_m1_error_rbf.csv'%(f, s)), delimiter=',')
        m2_data = np.genfromtxt(os.path.join(RESULTS_DIR, 'f%s_s%s_m2_error_rbf.csv'%(f, s)), delimiter=',')
        m3_data = np.genfromtxt(os.path.join(RESULTS_DIR, 'f%s_s%s_m3_error_rbf.csv'%(f, s)), delimiter=',')
        m4_data = np.genfromtxt(os.path.join(RESULTS_DIR, 'f%s_s%s_m4_error_rbf.csv'%(f, s)), delimiter=',')
        
        m2_data = np.reshape(m2_data, (len(m2_data), 1))
        m3_data = np.reshape(m3_data, (len(m3_data), 1))
        m4_data = np.reshape(m4_data, (len(m4_data), 1))
        
        m1_processed = np.full(m2_data.shape, np.mean(m1_data[:, 1]))
        
        graph_data = np.hstack((m1_processed, m2_data, m3_data, m4_data))
        
        rbf_graph_data.append(graph_data)

# Average data & make graph 
total = None 
for f in range(len(folds)):
    if f == 0:
        total = rbf_graph_data[f]
    else:
        total += rbf_graph_data[f]

rbf_graph_data_avg = total/len(folds)

plt.plot(x_ticks, rbf_graph_data_avg[:, :1], linewidth=2.0, color='c', label='sGP')
plt.plot(x_ticks, rbf_graph_data_avg[:, 1:2], linewidth=2.0, color='m', label='Error Sampling')
plt.plot(x_ticks, rbf_graph_data_avg[:, 2:3], linewidth=2.0, color='g', label='Uncertainty Sampling')
plt.plot(x_ticks, rbf_graph_data_avg[:, 3:], linewidth=2.0, color='gray', label='Random Sampling')
plt.legend(loc=1)
plt.xlabel('Number of Training Data')
plt.ylabel('MSE')
plt.grid(True)
plt.ylim(3,5.5)
if len(folds) == 1:
    plt.title('MSE v. Training Data: sGP RBF Fold %s'%(folds[0]))
    plt.savefig(os.path.join(GRAPH_FOLDER_DIR, 'rbf_fold_%s'%(folds[0])), dpi=300)
    plt.clf()
else:
    plt.title('MSE v. Training Data: sGP RBF Folds %s-%s'%(folds[0], folds[-1]))
    plt.savefig(os.path.join(GRAPH_FOLDER_DIR, 'rbf_fold_%s-%s'%(folds[0], folds[-1])), dpi=300)
    plt.clf()

ard_graph_data = []
# For ARD: 
for f in folds:
    for s in [1, 2, 3, 4, 5]:
        # Load CSV data for each model 
        m1_data = np.genfromtxt(os.path.join(RESULTS_DIR, 'f%s_s%s_m1_error_ard.csv'%(f, s)), delimiter=',')
        m2_data = np.genfromtxt(os.path.join(RESULTS_DIR, 'f%s_s%s_m2_error_ard.csv'%(f, s)), delimiter=',')
        m3_data = np.genfromtxt(os.path.join(RESULTS_DIR, 'f%s_s%s_m3_error_ard.csv'%(f, s)), delimiter=',')
        m4_data = np.genfromtxt(os.path.join(RESULTS_DIR, 'f%s_s%s_m4_error_ard.csv'%(f, s)), delimiter=',')
        
        m2_data = np.reshape(m2_data, (len(m2_data), 1))
        m3_data = np.reshape(m3_data, (len(m3_data), 1))
        m4_data = np.reshape(m4_data, (len(m4_data), 1))
        
        m1_processed = np.full(m2_data.shape, np.mean(m1_data[:, 1]))
        
        graph_data = np.hstack((m1_processed, m2_data, m3_data, m4_data))
        
        ard_graph_data.append(graph_data)

# Average data & make graph 
total = None 
for f in range(len(folds)):
    if f == 0:
        total = ard_graph_data[f]
    else:
        total += ard_graph_data[f]

ard_graph_data_avg = total/len(folds)

plt.plot(x_ticks, ard_graph_data_avg[:, :1], linewidth=2.0, color='c', label='sGP')
plt.plot(x_ticks, ard_graph_data_avg[:, 1:2], linewidth=2.0, color='m', label='Error Sampling')
plt.plot(x_ticks, ard_graph_data_avg[:, 2:3], linewidth=2.0, color='g', label='Uncertainty Sampling')
plt.plot(x_ticks, ard_graph_data_avg[:, 3:], linewidth=2.0, color='gray', label='Random Sampling')
plt.legend(loc=1)
plt.xlabel('Number of Training Data')
plt.ylabel('MSE')
plt.grid(True)
plt.ylim(3,5.5)
if len(folds) == 1:
    plt.title('MSE v. Training Data: sGP ARD Fold %s'%(folds[0]))
    plt.savefig(os.path.join(GRAPH_FOLDER_DIR, 'ard_fold_%s'%(folds[0])), dpi=300)
    plt.clf()
else:
    plt.title('MSE v. Training Data: sGP ARD Folds %s-%s'%(folds[0], folds[-1]))
    plt.savefig(os.path.join(GRAPH_FOLDER_DIR, 'ard_fold_%s-%s'%(folds[0], folds[-1])), dpi=300)
    plt.clf()
