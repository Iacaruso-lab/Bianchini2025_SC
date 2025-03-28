

#%% Import packages

import mat73
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import random

import more_itertools

sys.path.insert(0, 'Z:\\home\\shared\\Gaia\\Coliseum\\Delays\\paper_code\\Analysis\\helper_functions')

from functions_analysis import shiftA,doPCA_forSVM, run_clf, makeBins_SC, getBinIDs, run_clf_kfold,shift_sum

data_path = 'Z:\\home\\shared\\Gaia\\Coliseum\\Delays\\paper_code\\Datasets\\'
this_path = ''.join([data_path,'decoder_datasets\\clf_1D\\'])
#%% load dataset

file= ''.join([data_path,'neurons_datasets\\delay_tuning_dataset.mat'])

data_dict = mat73.loadmat(file)
DAT=data_dict['merged_dataset']
#check keys available
print(DAT.keys())

#%% Extract from dataset

spikes_tot=DAT['spikes']
trials=DAT['trials']

animal_IDs = DAT['animal_ID']
experiment_IDs = DAT['experiment_ID']
coord3D=DAT['coord3D']
coord3D_lab=['AP','depth_in_brain','ML']

AP_lim=DAT['AP_lim']
ML_lim=DAT['ML_lim']
depth_lim=DAT['depth_lim']


#%% try something different 

actual_lengthML = ML_lim[1] - ML_lim[0]
actual_lengthAP = AP_lim[1] - AP_lim[0]
actual_lengthdepth = depth_lim[1] - depth_lim[0]
# maybe it's better to take the actual length, so subtract the minimum of the range
ML_norm = coord3D[:,2] - ML_lim[0]
AP_norm = coord3D[:,0] - AP_lim[0]
depth_norm = coord3D[:,1] - depth_lim[0]

#%% there are some nans, because I dont have the positions of those, so excldue them 

good_pos = np.argwhere(~np.isnan(ML_norm))
AP_norm = AP_norm[good_pos]
ML_norm = ML_norm[good_pos]
depth_norm = depth_norm[good_pos]
spikes_tot = spikes_tot[good_pos[:,0],:,:]

animal_IDs = animal_IDs[good_pos]
experiment_IDs = experiment_IDs[good_pos]

#%% invert the ML axis first

max_val = actual_lengthML
min_val = 0 
new_ML = np.array([max_val - val + min_val for val in ML_norm])
ML_norm = new_ML

#%% divide it in bins - let's do bins of 400um 

possible_axis = ['AP','ML','depth']
a=int(list(sys.argv)[1])
axis = possible_axis[a]
print(axis)
#%%

if axis == 'ML':# add the extremes ML
    extremes = np.array([0,actual_lengthML])
    norm_coord = ML_norm
    actual_length = actual_lengthML
    
elif axis == 'AP':
    # add the extremes AP
    extremes = np.array([0,actual_lengthAP])
    norm_coord = AP_norm
    actual_length = actual_lengthAP

elif axis == 'depth':
    # add the extremes
    extremes = np.array([0,actual_lengthdepth])
    norm_coord = depth_norm
    actual_length = actual_lengthdepth

norm_coord2 = np.concatenate([norm_coord[:,0],extremes])

# divide the axis in 50um bins
# ask how many bins you need to have 50 micron
accuracy_range = 50
n_bins = int(np.ceil(actual_length/accuracy_range))
ids,bin_edges  = makeBins_SC(norm_coord2,n_bins)

#exclude the extrimes
ids = ids[:-2]
how_many = []
for i in range(1,n_bins+1):
    how_many.append(ids[ids==i].shape)

print(how_many)
how_many = np.array(how_many)

# what ranges do you want to try?
range_length = np.arange(250,750,50)

# the actual steps will be a multiple of this - each bin is 50um
actual_steps = (range_length/accuracy_range).tolist()
    
# classifier code 

# get spikes and labels in the right shape 
all_spikes_sum, all_labels_sum = shift_sum(spikes_tot,trials,sub_mean=0)

# get spikes and labels in the right shape 
all_spikes_shift, all_labels = shiftA(spikes_tot,trials,'all')

# get the one you want 
to_keep = spikes_tot.shape[1]
all_spikes = np.concatenate([all_spikes_sum[:,to_keep:-50,:],all_spikes_shift[:,to_keep-100:,:]],axis=1)

# set number of iterations
n_iter = 20
k_fold = 5
do_PCA = 0
#%%
# here I decide how many neurons to take
neurons =[25,50,75,100]
n_output = int(all_spikes.shape[1]/k_fold)
for n_neurons in neurons:

    # here I decide the bin size
    for steps in actual_steps:
        
        step_size = int(steps*accuracy_range)
        bin_ranges = np.arange(1,n_bins,steps) # these are the limits of the bins
        tot_bins = bin_ranges.shape[0] # how many bins will I have?    

        # initiate variables that will be saved
        predicted = np.zeros((tot_bins,n_iter,k_fold,n_output))
        y_test = np.zeros((tot_bins,n_iter,k_fold,n_output))

        specs_bins = np.zeros((tot_bins,4))
        specs_neurons = np.zeros((tot_bins,n_iter,n_neurons,2))
        
        # initiate empty list
        curr_spikes=[]
        for t in range(tot_bins): #iterate through all the bins
            # which bin should I extract?
            which_bins = np.arange(bin_ranges[t], n_bins+1 if t == tot_bins-1 else bin_ranges[t+1])
            
            # extract all the ids of the neurons that you will use
            curr_spikes = np.concatenate([np.where(ids == b)[0] for b in which_bins])

            # things to save - length of bins in position 0 and number of neurons per bin in poisition 1
            specs_bins[t,0] = which_bins.shape[0]*accuracy_range # length in um of the bin
            specs_bins[t,1] = curr_spikes.shape[0] # number of neurons in the bin
            specs_bins[t,2] = np.unique(animal_IDs[curr_spikes]).shape[0] # animal recorded in this bin
            specs_bins[t,3] = np.unique(experiment_IDs[curr_spikes]).shape[0] # number of experiments in this bin
            
            # if I actually have enough neurons in the bin do the computation :D 
            if specs_bins[t,1]>n_neurons:
                
                for n in tqdm(range(n_iter)): # select different random neurons at each iteration
                    
                    # get the ids you will use in this iteration
                    ids2 = random.sample(list(curr_spikes),n_neurons) 
                
                    #select the neurons
                    selected=all_spikes[ids2,:,:]
                    
                    # reshape it and do PCA
                    if do_PCA ==1:
                        scores = doPCA_forSVM(selected)
                    else:
                        scores = selected
                        
                    # run CLF
                    n_comp = scores.shape[-1]
                    predicted[t,n,:,:], y_test[t,n,:,:] = run_clf_kfold(n_comp, scores, all_labels,k_fold = k_fold, classifier ='SVM')
                        
                    # the last thing for you to save is the ids of the neurons and positions
                    specs_neurons[t,n,:,0] = ids2
                    specs_neurons[t,n,:,1] = norm_coord[ids2].reshape(-1)

        # what do i want to save?             
        np.save(''.join([this_path,'{axis}_clf_PRED_n{n_neurons}_step{step_size}_SVM_linear.npy']),predicted) # the predicted labels
        np.save(''.join([this_path,'{axis}_clf_TEST_n{n_neurons}_step{step_size}_SVM_linear.npy']),y_test) # the test labels   
        np.save(''.join([this_path,'{axis}_clf_IDS_n{n_neurons}_step{step_size}_SVM_linear.npy']),specs_neurons) # the ids of the neurons
        np.save(''.join([this_path,'{axis}_clf_BINS_n{n_neurons}_step{step_size}_SVM_linear.npy']),specs_bins) # the ids of the neurons # the number of session and animals in the bin 

neurons =[25,50,75,100]
for n_neurons in neurons:
# also do the random ones

    # initiate variables that will be saved
    predicted = np.zeros((n_iter,k_fold,n_output))
    y_test = np.zeros((n_iter,k_fold,n_output))

    specs_neurons = np.zeros((n_iter,n_neurons,2))

    for n in tqdm(range(n_iter)): # select increasing amount of neurons at each iteration

        curr_spikes=np.arange(all_spikes.shape[0]).tolist()
        # get the ids you will use in this iteration
        ids = random.sample(list(curr_spikes),n_neurons) 
    
        #select the neurons
        selected=all_spikes[ids,:,:]
        
        # reshape it and do PCA
        if do_PCA ==1:
            scores = doPCA_forSVM(selected)
        else:
            scores = selected
            
        # run CLF
        n_comp = scores.shape[-1]
        predicted[n,:,:], y_test[n,:,:] = run_clf_kfold(n_comp, scores, all_labels,k_fold = k_fold, classifier ='SVM')
                        
        # the last thing for you to save is the ids of the neurons and positions
        specs_neurons[n,:,0] = ids
        specs_neurons[n,:,1] = norm_coord[ids].reshape(-1)
        
    # what do i want to save?             
    np.save(''.join([this_path,'{axis}_clf_PRED_n{n_neurons}_random_SVM_linear.npy']),predicted) # the predicted labels
    np.save(''.join([this_path,'{axis}_clf_TEST_n{n_neurons}_random_SVM_linear.npy']),y_test) # the test labels   
    np.save(''.join([this_path,'{axis}_clf_IDS_n{n_neurons}_random_SVM_linear.npy']),specs_neurons) # the ids of the neurons

