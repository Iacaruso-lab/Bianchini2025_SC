#%% Import packages

import mat73
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
import random

sys.path.insert(0, 'Z:\\home\\shared\\Gaia\\Coliseum\\Delays\\paper_code\\Analysis\\helper_functions')
#from functions_analysis import *
from functions_analysis import shiftA,doPCA_forSVM, run_clf_kfold,shift_sum
data_path = 'Z:\\home\\shared\\Gaia\\Coliseum\\Delays\\paper_code\\Datasets\\'
this_path = ''.join([data_path,'decoder_datasets\\clf_increasing_N_RFC_linear\\'])

#%% load the dataset
file= ''.join([data_path,'neurons_datasets\\delay_tuning_dataset.mat'])
data_dict = mat73.loadmat(file)
DAT=data_dict['delay_tuning_dataset']

#check keys available
print(DAT.keys())

#%% Extract from dataset

spikes_tot=DAT['spikes']
trials=DAT['trials']

#%% classifier code 

# get spikes and labels in the right shape 
all_spikes_sum, all_labels_sum = shift_sum(spikes_tot,trials,sub_mean=1)

# get spikes and labels in the right shape 
all_spikes_shift, all_labels = shiftA(spikes_tot,trials,'all')

# get the one you want 
to_keep = spikes_tot.shape[1]
all_spikes = np.concatenate([all_spikes_sum[:,to_keep:-50,:],all_spikes_shift[:,to_keep-100:,:]],axis=1)

# set number of iterations
n_iter = 20
do_PCA = 0
k_fold = 5

curr_spikes=np.arange(all_spikes.shape[0]).tolist()
max_n = len(curr_spikes)
n_output = int(all_spikes.shape[1]/k_fold)
print(all_spikes.shape[2])
# let's run the bins in parallel # here I decide the bin size
rep_n=int(list(sys.argv)[1])
print(rep_n)

# Create an array from 0 to max_n with a step of 50
this_n = np.arange(0, max_n, 25)

# Ensure max_n is included
if this_n[-1] != max_n:
    this_n = np.append(this_n, max_n-1)

only_random = 0
if only_random == 1:
    print('only random')
else:
    # initiate variables that will be saved
    predicted = np.zeros((n_iter,max_n,k_fold,n_output))
    y_test = np.zeros((n_iter,max_n,k_fold,n_output))
    #specs_neurons = np.zeros((n_iter,max_n,2))


    for n in tqdm(this_n):#tqdm(range(len(curr_spikes))):       
                
        # get the ids you will use in this iteration
        ids = random.sample(list(curr_spikes),n+1) 

        #select the neurons
        selected=all_spikes[ids,:,:]

        # reshape it and do PCA
        if do_PCA ==1:
            scores = doPCA_forSVM(selected)
        else:
            scores = selected
        
        # run CLF
        n_comp = scores.shape[-1]
        predicted[rep_n,n,:,:], y_test[rep_n,n,:,:] = run_clf_kfold(n_comp, scores, all_labels,k_fold = k_fold, classifier ='SVM')
        #predicted[rep_n,n,:], y_test[rep_n,n,:], CLF_scores = run_clf(n_comp, scores, all_labels,classifier = 'RFC')

    # save the results
    # what do i want to save? 
    file =''.join([this_path,f'PRED_rep{rep_n}_SVM.npy'])             
    np.save(file,predicted) # the predicted labels

    file =''.join([this_path,f'TEST_rep{rep_n}_SVM.npy'])             
    np.save(file,y_test) # the test labels   

#%% do it for random labels

# initiate variables that will be saved
predicted = np.zeros((n_iter,max_n,k_fold,n_output))
y_test = np.zeros((n_iter,max_n,k_fold,n_output))
for n in tqdm(this_n):#tqdm(range(len(curr_spikes))):   

    # shuffle the labels
    shuffle_all_labels=np.array(random.sample(all_labels.tolist(),all_labels.shape[0])) # shuffle the labels here        
    # get the ids you will use in this iteration
    ids = random.sample(list(curr_spikes),n+1) 

    #select the neurons
    selected=all_spikes[ids,:,:]

    # reshape it and do PCA
    if do_PCA ==1:
        scores = doPCA_forSVM(selected)
    else:
        scores = selected
    
    # run CLF
    n_comp = scores.shape[-1]
    predicted[rep_n,n,:,:], y_test[rep_n,n,:,:] = run_clf_kfold(n_comp, scores, shuffle_all_labels,k_fold = k_fold, classifier ='SVM')

# save the results
# what do i want to save? 
file =''.join([this_path,f'PRED_rep{rep_n}_random_SVM.npy'])             
np.save(file,predicted) # the predicted labels

file =''.join([this_path,f'TEST_rep{rep_n}_random_SVM.npy'])             
np.save(file,y_test) # the test labels