#%% Import packages

import mat73
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import random

sys.path.insert(0, 'Z:\\home\\shared\\Gaia\\Coliseum\\Delays\\paper_code\\Analysis\\helper_functions')
#from functions_analysis import *
from functions_analysis import shiftA,doPCA_forSVM, run_clf_kfold
data_path = 'Z:\\home\\shared\\Gaia\\Coliseum\\Delays\\paper_code\\Datasets\\'
this_path = ''.join([data_path,'decoder_datasets\\clf_increasing_N_SVM\\'])

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
all_spikes, all_labels = shiftA(spikes_tot,trials,'all')

# set number of iterations
n_iter = 200
do_PCA = 0
k_fold = 5

curr_spikes=np.arange(all_spikes.shape[0]).tolist()
max_n = 201
n_output = int(all_spikes.shape[1]/k_fold)
print(all_spikes.shape[2])

# let's run the bins in parallel # here I decide the bin size
predicted = np.zeros((n_iter,max_n,k_fold,n_output))
y_test = np.zeros((n_iter,max_n,k_fold,n_output))
for rep_n in tqdm(range(n_iter)):

    for n in tqdm(range(max_n)):                       
        # get the ids you will use in this iteration
        ids = random.sample(list(curr_spikes),n+1) 

        #select the neurons
        scores=all_spikes[ids,:,:]

        # run CLF
        n_comp = scores.shape[-1]
        predicted[rep_n,n,:,:], y_test[rep_n,n,:,:] = run_clf_kfold(n_comp, scores, all_labels,k_fold = k_fold, classifier ='SVM')
   
# what do i want to save? 
file =''.join([this_path,'PRED_first200neurons_SVM.npy'])             
np.save(file,predicted) # the predicted labels

file =''.join([this_path,'TEST_first200neurons_SVM.npy'])             
np.save(file,y_test) # the test labels 