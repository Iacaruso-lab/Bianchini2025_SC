#%% Import packages

import mat73
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import random
import pickle
import more_itertools

sys.path.insert(0, 'Z:\\home\\shared\\Gaia\\Coliseum\\Delays\\paper_code\\Analysis\\helper_functions')
#from functions_analysis import *
from functions_analysis import shiftA,doPCA_forSVM, run_clf_kfold,shift_sum
data_path = 'Z:\\home\\shared\\Gaia\\Coliseum\\Delays\\paper_code\\Datasets\\'
this_path = ''.join([data_path,'decoder_datasets\\clf_increasing_N_SVM\\'])
                     
#%% does the decoder accuracy change based on position of the neurons

file= ''.join([data_path,'neurons_datasets\\delay_tuning_dataset.mat'])
data_dict = mat73.loadmat(file)
DAT=data_dict['delay_tuning_dataset']
#check keys available
print(DAT.keys())

#%% Extract from dataset

spikes_tot=DAT['spikes']
trials=DAT['trials']
experiment_ID = DAT['experiment_ID']
which_experiments = np.unique(experiment_ID)

#%% get delay neurons

peaks=DAT['peaks']

all_boot_aud=DAT['all_boot_aud']
all_boot_vis=DAT['all_boot_vis']

sig_del = []
for i in range(peaks.shape[0]):
    y = peaks[i,:-2]

    vis_FR = peaks[i,-2]
    aud_FR = peaks[i,-1]

    if vis_FR>aud_FR:
        boot_out = all_boot_vis[i,:]
    elif aud_FR>vis_FR:
        boot_out = all_boot_aud[i,:]
    
    pos_sig = np.argwhere(boot_out>0)

    if len(pos_sig)>0:
        sig_del.append(i)

sig_del = np.array(sig_del)

#%%

# get spikes and labels in the right shape 
all_spikes_sum, all_labels = shift_sum(spikes_tot,trials,sub_mean=1)

#%%
# get spikes and labels in the right shape 
all_spikes_shift, all_labels = shiftA(spikes_tot,trials,'all')

# get the one you want 
to_keep = spikes_tot.shape[1]
all_spikes = np.concatenate([all_spikes_sum[:,to_keep:-50,:],all_spikes_shift[:,to_keep-100:,:]],axis=1)

#%% only do this for delay neurons

my_spikes = np.squeeze(all_spikes[sig_del,:,:])
curr_spikes=np.arange(my_spikes.shape[0]).tolist()

# set number of iterations
n_iter = 20
do_PCA = 0
k_fold = 5
max_n = sig_del.shape[0]

# initiate variables that will be saved
n_output = int(all_spikes.shape[1]/k_fold)
predicted = np.zeros((n_iter,max_n,k_fold,n_output))
y_test = np.zeros((n_iter,max_n,k_fold,n_output))
# List to store ids
ids_tracking = []
rep_n=int(list(sys.argv)[1])
print(rep_n)

ids_tracking.append([])  # Initialize list for this repetition
for n in tqdm(range(len(curr_spikes))):       
            
    # get the ids you will use in this iteration
    ids = random.sample(list(curr_spikes),n+1) 
    ids_tracking.append(ids)  # Store the ids
    #select the neurons
    selected=my_spikes[ids,:,:]

    # reshape it and do PCA
    if do_PCA ==1:
        scores = doPCA_forSVM(selected)
    else:
        scores = selected
    
    # run CLF
    n_comp = scores.shape[-1]
    predicted[rep_n,n,:,:], y_test[rep_n,n,:,:] = run_clf_kfold(n_comp, scores, all_labels,k_fold = k_fold, classifier ='SVM')

# what do i want to save? 
file =''.join([this_path,f'PRED_rep{i}_linear_SVM_delay.npy'])             
np.save(file,predicted) # the predicted labels

file =''.join([this_path,f'TEST_rep{i}_linear_SVM_delay.npy'])           
np.save(file,y_test) # the test labels  

#%% and let's calculate also the non linear (OBSERVED)

# get spikes and labels in the right shape 
all_spikes, all_labels = shiftA(spikes_tot,trials,'all')

my_spikes = np.squeeze(all_spikes[sig_del,:,:])
curr_spikes=np.arange(my_spikes.shape[0]).tolist()

# set number of iterations
n_iter = 20
do_PCA = 0
k_fold = 5
max_n = sig_del.shape[0]

# initiate variables that will be saved
n_output = int(all_spikes.shape[1]/k_fold)
predicted = np.zeros((n_iter,max_n,k_fold,n_output))
y_test = np.zeros((n_iter,max_n,k_fold,n_output))
# List to store ids
ids_tracking = []
rep_n=int(list(sys.argv)[1])
print(rep_n)

ids_tracking.append([])  # Initialize list for this repetition
for n in tqdm(range(len(curr_spikes))):       
            
    # get the ids you will use in this iteration
    ids = random.sample(list(curr_spikes),n+1) 
    #ids_tracking[rep_n].append(ids)  # Store the ids
    #select the neurons
    selected=my_spikes[ids,:,:]

    # reshape it and do PCA
    if do_PCA ==1:
        scores = doPCA_forSVM(selected)
    else:
        scores = selected
    
    # run CLF
    n_comp = scores.shape[-1]
    predicted[rep_n,n,:,:], y_test[rep_n,n,:,:] = run_clf_kfold(n_comp, scores, all_labels,k_fold = k_fold, classifier ='SVM')

# what do i want to save? 
file =''.join([this_path,f'PRED_rep{i}_SVM_delay.npy'])             
np.save(file,predicted) # the predicted labels

file =''.join([this_path,f'TEST_rep{i}_SVM_delay.npy'])           
np.save(file,y_test) # the test labels  