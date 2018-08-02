# Processing ML Experiment Requests - Histogram 

# Import libraries 
import os
import numpy as np 
import pathlib
import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter

plt.rcParams['axes.facecolor'] = 'lightgray'
plt.rcParams['font.family'] = 'serif'

## PREPARE DATA  
print('----- PREPARING DATA -----')

# Define directories
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, 'ml_exp_results') 

GRAPH_FOLDER_DIR = os.path.join(RESULTS_DIR, 'graphs')
pathlib.Path(GRAPH_FOLDER_DIR).mkdir(parents=True, exist_ok=True) 

threshold_vals = [0, 1, 1.5, 2, 2.5, 3, 3.5, 4]

for kern in ['rbf', 'ard']:
    for t in threshold_vals:
        g1_REQ_VEC_CSV_DIR = os.path.join(RESULTS_DIR, 't%s_pGP_%s_g1_req_vec.csv'%(t, kern))
        g1_req_vec_data = np.genfromtxt(g1_REQ_VEC_CSV_DIR, delimiter=',')
        g2_REQ_VEC_CSV_DIR = os.path.join(RESULTS_DIR, 't%s_pGP_%s_g2_req_vec.csv'%(t, kern))
        g2_req_vec_data = np.genfromtxt(g2_REQ_VEC_CSV_DIR, delimiter=',')
        g3_REQ_VEC_CSV_DIR = os.path.join(RESULTS_DIR, 't%s_pGP_%s_g3_req_vec.csv'%(t, kern))
        g3_req_vec_data = np.genfromtxt(g3_REQ_VEC_CSV_DIR, delimiter=',')
        
        g1_y = np.sum(g1_req_vec_data[:,1:], axis=0)/25*100
        g2_y = np.sum(g2_req_vec_data[:,1:], axis=0)/27*100
        g3_y = np.sum(g3_req_vec_data[:,1:], axis=0)/48*100
        
        x = np.arange(1,len(g1_y)+1)
        
        ax = plt.subplot(111)
        
        ax.bar(x-0.3, g1_y, width=0.3, color='k', align='center', label='Group 1')
        ax.bar(x, g2_y, width=0.3, color='gray', align='center', label='Group 2')
        ax.bar(x+0.3, g3_y, width=0.3, color='c', align='center', label='Group 3')
        
        formatter = FuncFormatter(lambda y, pos: "%d%%" % (y))
        ax.yaxis.set_major_formatter(formatter)

        plt.legend(loc=1)
        
        plt.xlabel('Visit Number')
        plt.ylabel('Percentage of Requests')
        plt.grid(True)
        
        plt.ylim(0,105) 
        
        plt.title('Percentage of Label Requests per Group: Threshold=%s Kernel=%s'%(t, kern))
        plt.savefig(os.path.join(GRAPH_FOLDER_DIR, 't%s_pGP_%s_req_graph.png'%(t, kern)), dpi=300)
        
        plt.clf()