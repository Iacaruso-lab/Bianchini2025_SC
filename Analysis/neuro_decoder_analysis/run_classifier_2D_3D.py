# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 14:16:49 2023

@author: bianchg
"""

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
from functions_analysis import shiftA,doPCA_forSVM, run_clf, makeBins_SC, getBinIDs,run_clf_kfold

data_path = 'Z:\\home\\shared\\Gaia\\Coliseum\\Delays\\paper_code\\Datasets\\'
this_path = ''.join([data_path,'decoder_datasets\\clf_2D_3D\\'])

#%% load the dataset 

file= ''.join([data_path,'neurons_datasets\\delay_tuning_dataset.mat'])

data_dict = mat73.loadmat(file)
DAT=data_dict['merged_dataset']
#check keys available
print(DAT.keys())

#%% Extract from dataset

spikes_tot=DAT['spikes']
trials=DAT['trials']

coord3D=DAT['coord3D']
coord3D_lab=['AP','depth_in_brain','ML']
animal_IDs = DAT['animal_ID']
experiment_IDs = DAT['experiment_ID']

AP_lim=DAT['AP_lim']
ML_lim=DAT['ML_lim']
depth_lim=DAT['depth_lim']
peaks=DAT['peaks']
all_boot_aud=DAT['all_boot_aud']
all_boot_vis=DAT['all_boot_vis']

#normalize values
#%% try something different 

actual_lengthML = ML_lim[1] - ML_lim[0]
actual_lengthAP = AP_lim[1] - AP_lim[0]
actual_lengthdepth = depth_lim[1] - depth_lim[0]

# maybe it's better to take the actual length, so subtract the minimum of the range
ML_norm = coord3D[:,2] - ML_lim[0]
AP_norm = coord3D[:,0] - AP_lim[0]
depth_norm = coord3D[:,1] - depth_lim[0]

#%% there are some nans, because I dont have the positions of those, so excldue them 

good_pos = np.argwhere(~np.isnan(depth_norm))
AP_norm = AP_norm[good_pos]
ML_norm = ML_norm[good_pos]
depth_norm = depth_norm[good_pos]
spikes_tot = spikes_tot[good_pos[:,0],:,:]

animal_IDs = animal_IDs[good_pos]
experiment_IDs = experiment_IDs[good_pos]

modality = DAT['modality'][good_pos]
peaks=np.squeeze(DAT['peaks'][good_pos])
all_boot_aud=np.squeeze(DAT['all_boot_aud'][good_pos])
all_boot_vis=np.squeeze(DAT['all_boot_vis'][good_pos])

#%% invert the ML axis first

max_val = actual_lengthML
min_val = 0 
new_ML = np.array([max_val - val + min_val for val in ML_norm])
ML_norm = new_ML
#%% divide it in bins

extremes = np.array([0,actual_lengthAP])
AP_norm2 = np.concatenate([AP_norm[:,0],extremes])

extremes = np.array([0,actual_lengthML])
ML_norm2 = np.concatenate([ML_norm[:,0],extremes])

extremes = np.array([0,actual_lengthdepth])
depth_norm2 = np.concatenate([depth_norm[:,0],extremes])

#%% do it both for 2d bins and 3d bins
skip = 0
n_bins_tot=2
id_AP,edges_AP  = makeBins_SC(AP_norm2,n_bins_tot)
id_ML,edges_ML = makeBins_SC(ML_norm2,n_bins_tot)
id_depth,edges_depth = makeBins_SC(depth_norm2,n_bins_tot)

id_AP = id_AP[:-2]
id_ML = id_ML[:-2]
id_depth = id_depth[:-2]

# I want to save the length of each bin
length_bins = [edges_AP[1],edges_ML[1]]
#%%
# after creating the bins create the combinations and give to each neuron the id of the bin it is in getBinIDs
dim = [2]
for n_var in dim: # set if you want to do it in 3d or 2d

    if n_var == 2:
        var1,var2 = id_ML,id_AP
        groups,lengths,ID_neurons,bins = getBinIDs(n_var,var1,var2,n_bins_tot)
        label_groups = ['MA','MP','LA','LP']
    elif n_var ==3:
        var1,var2,var3 = id_ML,id_AP,id_depth
        groups,lengths,ID_neurons,bins = getBinIDs(n_var,var1,var2,n_bins_tot,var3)
        label_groups = ['sMA','dMA','sMP','dMP','sLA','dLA','sLP','dLP']

    print(lengths)
    lengths = np.array(lengths)

    # extract the number of vis, aud, multi neurons in each bin
    n_modalities = np.zeros((bins,np.unique(modality).shape[0]))

    for b in range(1,bins+1): #iterate through all the bins
        temp1 = np.where(ID_neurons == b)[0] # get the neurons in that bin    
        #get how many neurons of each modality are in that bin
        count_me=np.unique(modality[temp1],return_counts=True)
        n_modalities[b-1,count_me[0].astype(int)] = count_me[1]

    # extract the number of delay neurons and type of delays in each bin
    # first what are the locations of the delay neurons in the big matrix?

    sig_del = []
    which_tr = []
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
            tr = pos_sig[np.argmax(y[pos_sig])]
            which_tr.append(tr) 

    sig_del = np.array(sig_del)
    all_peaks=np.array(which_tr)[:,0]

    n_delay = np.zeros((bins,1))
    type_delay = np.zeros((bins,np.unique(all_peaks).shape[0]))
    for b in range(1,bins+1): #iterate through all the bins
        temp1 = np.where(ID_neurons == b)[0] # get the neurons in that bin  
        which_n = np.intersect1d(sig_del,temp1)
        positions = np.where(np.isin(sig_del, which_n))[0]
        n_delay[b-1] = which_n.shape[0]
        #get how the number of neurons of each type of delay are in that bin
        count_me=np.unique(all_peaks[positions],return_counts=True)
        type_delay[b-1,count_me[0].astype(int)] = count_me[1]


    # save some specs

    SPECS_dict = dict(groups = groups,axes_groups = label_groups,n_neurons = lengths,n_modalities = n_modalities,n_delay = n_delay,type_delay = type_delay)
    # SVM code 

    # here I decide how many neurons to take
    neurons = [25,50,75,100]
    real_n=int(list(sys.argv)[1])
    n_neurons = neurons[real_n]

    # get spikes and labels in the right shape 
    all_spikes, all_labels = shiftA(spikes_tot,trials,'all')

    # Reshape the resulting array
    all_spikes = all_spikes.reshape(all_spikes.shape[0], -1, all_spikes.shape[2])
    all_spikes = np.where(all_spikes < 0, 0, all_spikes) #to avoid having negative values

    # set number of iterations
    n_iter = 20
    do_PCA = 0

    # initiate variables that will be saved
    k_fold = 5
    n_output = int(all_spikes.shape[1]/k_fold)
    predicted = np.zeros((bins,n_iter,k_fold,n_output))
    y_test = np.zeros((bins,n_iter,k_fold,n_output))
    
    specs_bins = np.zeros((bins,4))
    specs_neurons = np.zeros((bins,n_iter,n_neurons,4))
    ids_tracking = []
    # initiate empty list
    curr_spikes=[]
    for b in range(1,bins+1): #iterate through all the bins
        
        # extract all the ids of the neurons that you will use
        curr_spikes = np.concatenate(np.where(ID_neurons == b))
        
        # things to save - length of bins in position 0 and number of neurons per bin in poisition 1
        #specs_bins[t,0] =  # length in um of the bin
        specs_bins[b-1,1] = curr_spikes.shape[0] # number of neurons in the bin
        specs_bins[b-1,2] = np.unique(animal_IDs[curr_spikes]).shape[0] # animal recorded in this bin
        specs_bins[b-1,3] = np.unique(experiment_IDs[curr_spikes]).shape[0] # number of experiments in this bin
        
        # if I actually have enough neurons in the bin do the computation :D 
        if specs_bins[b-1,1]>n_neurons:
            
            for n in tqdm(range(n_iter)): # select increasing amount of neurons at each iteration
                
                # get the ids you will use in this iteration
                ids = random.sample(list(curr_spikes),n_neurons) 
                #ids_tracking[b].append(ids)  # Store the ids
                #select the neurons
                selected=all_spikes[ids,:,:]
                
                # reshape it and do PCA
                if do_PCA ==1:
                    scores = doPCA_forSVM(selected)
                else:
                    scores = selected
                    
                # run CLF
                n_comp = scores.shape[-1]
                predicted[b-1,n,:,:], y_test[b-1,n,:,:] = run_clf_kfold(n_comp, scores, all_labels,k_fold = k_fold, classifier ='SVM')
    
                # the last thing for you to save is the ids of the neurons and positions
                specs_neurons[b-1,n,:,0] = ids
                specs_neurons[b-1,n,:,1] = ML_norm[ids].reshape(-1)
                specs_neurons[b-1,n,:,1] = AP_norm[ids].reshape(-1)
                specs_neurons[b-1,n,:,1] = depth_norm[ids].reshape(-1)
                

    np.save(''.join([this_path,'PRED_n{n_neurons}_{n_var}_SVM.npy']),predicted)  # the predicted labels
    np.save(''.join([this_path,'TEST_n{n_neurons}_{n_var}_SVM.npy']),y_test) # the test labels  
    np.save(''.join([this_path,'IDS_n{n_neurons}_{n_var}_SVM.npy']),specs_neurons) # the ids of the neurons
    np.save(''.join([this_path,'BINS_n{n_neurons}_{n_var}_SVM.npy']),specs_bins) # the ids of the neurons # the number of session and animals in the bin 
    np.save(''.join([this_path,'n_modalities_n{n_neurons}_{n_var}_SVM.npy']),n_modalities) # the ids of the neurons # the number of session and animals in the bin 
    np.save(''.join([this_path,'n_delay_n{n_neurons}_{n_var}_SVM.npy']),n_delay) # the ids of the neurons # the number of session and animals in the bin 
    np.save(''.join([this_path,'type_delay_n{n_neurons}_{n_var}_SVM.npy']),type_delay) # the ids of the neurons # the number of session and animals in the bin 
    # Save ids_tracking to a file
    with open(''.join([this_path,'ids_tracking_{n_neurons}_{n_var}_SVM.pkl']), 'wb') as f:
        pickle.dump(ids_tracking, f)
    
    import scipy.io as sio

    sio.savemat(''.join([this_path,'DICT_n{n_neurons}_{n_var}_SVM.mat']), {'my_dict': SPECS_dict})