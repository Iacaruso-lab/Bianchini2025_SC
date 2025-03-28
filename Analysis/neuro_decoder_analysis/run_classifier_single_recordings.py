

#%% run the decoder for single recordings 

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
from functions_analysis import shiftA,doPCA_forSVM, run_clf_kfold
data_path = 'Z:\\home\\shared\\Gaia\\Coliseum\\Delays\\paper_code\\Datasets\\'
this_path = ''.join([data_path,'decoder_datasets\\clf_single_rec\\'])

#%% load the dataset 
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

#%% classifier code 

# set number of iterations
n_iter = 20
do_PCA = 0
k_fold = 5

# let's run the bins in parallel # here I decide the bin size
rec_n=int(list(sys.argv)[1])
rec = which_experiments[rec_n].astype(int)
subset_neurons = np.argwhere(experiment_ID==rec)
max_n = subset_neurons.shape[0]

curr_spikes=np.arange(subset_neurons.shape[0]).tolist()
# get spikes and labels in the right shape 
#all_spikes, all_labels = shift_sum(spikes_tot,trials,sub_mean=1)
all_spikes, all_labels = shiftA(spikes_tot,trials,'all')
my_spikes = np.squeeze(all_spikes[subset_neurons,:,:])
# initiate variables that will be saved
n_output = int(all_spikes.shape[1]/k_fold)
predicted = np.zeros((n_iter,max_n,k_fold,n_output))
y_test = np.zeros((n_iter,max_n,k_fold,n_output))

# List to store ids
ids_tracking = []
for rep_n in tqdm(range(n_iter)):
    ids_tracking.append([])  # Initialize list for this repetition
    for n in tqdm(range(len(curr_spikes))):       
                
        # get the ids you will use in this iteration
        ids = random.sample(list(curr_spikes),n+1) 
        ids_tracking[rep_n].append(ids)  # Store the ids
        #select the neurons
        selected=my_spikes[ids,:,:]

        # reshape it and do PCA
        if do_PCA ==1:
            scores = doPCA_forSVM(selected)
        else:
            scores = selected
        
        # run CLF
        n_comp = scores.shape[-1]
        predicted[rep_n,n,:,:], y_test[rep_n,n,:,:] = run_clf_kfold(n_comp, scores, all_labels,k_fold = k_fold, classifier ='SVM') #choose RFC if want to run RFC
   
# save the results
# what do i want to save?             
np.save(''.join([this_path,f'rec{rec}_PRED_SVM.npy']),predicted) # the predicted labels
np.save(''.join([this_path,f'rec{rec}_TEST_SVM.npy']),y_test) # the test labels   

# Save ids_tracking to a file
with open(''.join([this_path,f'clf_single_rec/rec{rec}_ids_tracking_SVM.pkl']), 'wb') as f:
    pickle.dump(ids_tracking, f)

#%% do it for random labels

# initiate variables that will be saved
predicted = np.zeros((n_iter,max_n,k_fold,n_output))
y_test = np.zeros((n_iter,max_n,k_fold,n_output))

for rep_n in tqdm(range(n_iter)):

    for n in tqdm(range(len(curr_spikes))):       

        # shuffle the labels
        shuffle_all_labels=np.array(random.sample(all_labels.tolist(),all_labels.shape[0])) # shuffle the labels here        
        # get the ids you will use in this iteration
        ids = random.sample(list(curr_spikes),n+1) 

        #select the neurons
        selected=my_spikes[ids,:,:]

        # reshape it and do PCA
        if do_PCA ==1:
            scores = doPCA_forSVM(selected)
        else:
            scores = selected
        
        # run CLF
        n_comp = scores.shape[-1]
        predicted[rep_n,n,:,:], y_test[rep_n,n,:,:] = run_clf_kfold(n_comp, scores, shuffle_all_labels,k_fold = k_fold, classifier ='SVM') #choose RFC if want to run RFC

# save the results
np.save(''.join([this_path,f'rec{rec}_PRED_random_SVM.npy']),predicted) # the predicted labels
np.save(''.join([this_path,f'rec{rec}_TEST_random_SVM.npy']),y_test) # the test labels   