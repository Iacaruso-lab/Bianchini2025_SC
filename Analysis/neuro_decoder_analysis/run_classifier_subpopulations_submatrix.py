# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 15:17:37 2023

@author: bianchg
"""

#%% Population analysis on CAMP 

#%% Import packages

import mat73
import numpy as np
import sys
from tqdm import tqdm
import random

sys.path.insert(0, 'Z:\\home\\shared\\Gaia\\Coliseum\\Delays\\paper_code\\Analysis\\helper_functions')
#from functions_analysis import *
from functions_analysis import shiftA,doPCA_forSVM, run_clf_kfold
data_path = 'Z:\\home\\shared\\Gaia\\Coliseum\\Delays\\paper_code\\Datasets\\'
this_path = ''.join([data_path,'decoder_datasets\\clf_types_neurons_SVM\\'])


#%% load the dataset 
file= ''.join([data_path,'neurons_datasets\\delay_tuning_dataset.mat'])
data_dict = mat73.loadmat(file)
DAT=data_dict['delay_tuning_dataset']
#check keys available
print(DAT.keys())

#%% Extract from dataset

spikes_tot=DAT['spikes']
trials=DAT['trials']
resp=DAT['resp']
modality=DAT['modality']

#%% SVM with all neurons and A shifted and increasing n

# get spikes and labels in the right shape 
all_spikes2, all_labels2 = shiftA(spikes_tot,trials,'all')

#%%  let's pick a subset of spikes, to only have the AV delay trials
n_delays = 11
n_rep = 50
all_spikes = all_spikes2[:,:n_delays*n_rep,:]
all_labels = all_labels2[:n_delays*n_rep]

#%%
#
# set number of iterations
n_iter = 20
k_fold = 5
do_PCA = 0
n_modality = np.unique(modality)
n_output = int(all_spikes.shape[1]/k_fold)

predicted = np.zeros((len(n_modality),n_iter,3000,k_fold,n_output ))
y_test = np.zeros((len(n_modality),n_iter,3000,k_fold,n_output ))

m=int(list(sys.argv)[1])
print(m)
only_delay = 1
if only_delay ==1:
    print('only delay')
else:
    n_mod = 3 # only do vis, aud and multi all together

    if n_mod == 3:
        modality = np.array([min(value, 3) for value in modality])

    which_spikes = all_spikes[modality==m,:,:]
        
    if m == 0:
        max_neurons = 100
    else:
        max_neurons = which_spikes.shape[0]

    mx = int(m)
    seq=np.arange(which_spikes.shape[0]).tolist()
    this_n = np.arange(0, max_neurons, 5)

    # Ensure max_n is included
    if this_n[-1] != max_neurons:
        this_n = np.append(this_n, max_neurons-1)
        
    for t in tqdm(range(n_iter)):

        for n in tqdm(this_n): # select increasing amount of neurons at each iteration        
            
            ids=random.sample(seq, which_spikes.shape[0])
            #select the neurons
            selected=which_spikes[ids[:n+1],:,:]
            
            # reshape it and do PCA
            if do_PCA ==1:
                scores = doPCA_forSVM(selected)
            else:
                scores = selected# reshape it and do PCA
            
            # run SVM
            n_comp = scores.shape[-1]
            predicted[mx,t,n,:,:], y_test[mx,t,n,:,:]= run_clf_kfold(n_comp, scores, all_labels,k_fold = k_fold, classifier ='SVM')
    
    # what do i want to save? 
    file =''.join([this_path,'PRED_mod_%d' %m + '_SVM_small.npy'])             
    np.save(file,predicted) # the predicted labels

    file =''.join([this_path,'TEST_mod_%d' %m +'_SVM_small.npy'])             
    np.save(file,y_test) # the test labels  

#%% now calculate it also for delay neurons 

if m == 0:
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

    # set number of iterations
    n_neurons = sig_del.shape[0]
    max_neurons = sig_del.shape[0]

    mx = int(m)
    seq=np.arange(sig_del.shape[0]).tolist()
    this_n = np.arange(0, max_neurons, 5)

    # Ensure max_n is included
    if this_n[-1] != max_neurons:
        this_n = np.append(this_n, max_neurons-1)

    n_iter = 20
    n_rep = 50

    predicted = np.zeros((n_iter,n_neurons,k_fold,n_output))
    y_test = np.zeros((n_iter,n_neurons,k_fold,n_output))

    #for m in n_modality:
    which_spikes = all_spikes[sig_del,:,:]
        
    seq=np.arange(which_spikes.shape[0]).tolist()
        
    for t in tqdm(range(n_iter)):

        for n in tqdm(this_n): # select increasing amount of neurons at each iteration    
            
            ids=random.sample(seq, which_spikes.shape[0])
            #select the neurons
            selected=which_spikes[ids[:n+1],:,:]
            
            if do_PCA ==1:
                scores = doPCA_forSVM(selected)
            else:
                scores = selected# reshape it and do PCA
            
            # run SVM
            n_comp = scores.shape[-1]
            predicted[t,n,:,:], y_test[t,n,:,:]= run_clf_kfold(n_comp, scores, all_labels,k_fold = k_fold, classifier ='SVM')
    # what do i want to save? 
    file =''.join([this_path,'PRED_mod_delay_SVM_small.npy'])             
    np.save(file,predicted) # the predicted labels

    file =''.join([this_path,'TEST_mod_delay_SVM_small.npy'])             
    np.save(file,y_test) # the test labels   

#%% also let's calculate it for multisensory - delay

#%% now calculate it also for delay neurons 

if m == 0:
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

    # set number of iterations
    modality = np.array([min(value, 3) for value in modality])
    multi = np.argwhere(modality==3)
    
    which_neurons = np.setdiff1d(multi,sig_del)
    n_neurons = which_neurons.shape[0]
    max_neurons = which_neurons.shape[0]

    mx = int(m)
    seq=np.arange(which_neurons.shape[0]).tolist()
    this_n = np.arange(0, max_neurons, 5)

    # Ensure max_n is included
    if this_n[-1] != max_neurons:
        this_n = np.append(this_n, max_neurons-1)

    n_iter = 20
    n_rep = 50

    predicted = np.zeros((n_iter,n_neurons,k_fold,n_output))
    y_test = np.zeros((n_iter,n_neurons,k_fold,n_output))

    #for m in n_modality:
    which_spikes = all_spikes[which_neurons,:,:]
        
    seq=np.arange(which_spikes.shape[0]).tolist()
        
    for t in tqdm(range(n_iter)):

        for n in tqdm(this_n): # select increasing amount of neurons at each iteration    
            
            ids=random.sample(seq, which_spikes.shape[0])
            #select the neurons
            selected=which_spikes[ids[:n+1],:,:]
            
            if do_PCA ==1:
                scores = doPCA_forSVM(selected)
            else:
                scores = selected# reshape it and do PCA
            
            # run SVM
            n_comp = scores.shape[-1]
            predicted[t,n,:,:], y_test[t,n,:,:]= run_clf_kfold(n_comp, scores, all_labels,k_fold = k_fold, classifier ='SVM')

    # what do i want to save? 
    file =''.join([this_path,'PRED_mod_3NOdelay_SVM_small.npy'])             
    np.save(file,predicted) # the predicted labels

    file =''.join([this_path,'TEST_mod_3NOdelay_SVM_small.npy'])             
    np.save(file,y_test) # the test labels 