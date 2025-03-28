#%% import packages

import mat73
import numpy as np
import sys
from tqdm import tqdm
import random
import os
from scipy.interpolate import interp1d
from matplotlib.colors import LogNorm
import seaborn as sns

sys.path.insert(0, 'Z:\\home\\shared\\Gaia\\Coliseum\\Delays\\paper_code\\Analysis\\helper_functions')
from functions_analysis import shiftA_behav,doPCA_forSVM, run_clf_kfold

#%% load data
data_path = 'Z:\\home\\shared\\Gaia\\Coliseum\\Delays\\paper_code\\Datasets\\movement_control_datasets\\raw_data'
videoFolders = [os.path.join(dataPath, d) for d in os.listdir(dataPath) if os.path.isdir(os.path.join(dataPath, d))]  

save_path = 'Z:\\home\\shared\\Gaia\\Coliseum\\Delays\\paper_code\\Datasets\\movement_control_datasets\\decoder_analysis'

#%% load the data
 
i=int(list(sys.argv)[1])
print(i)

file=os.path.join(videoFolders[i],'dec_data.mat')
data_dict = mat73.loadmat(file)
DAT=data_dict['dec_data']

#check keys available
print(DAT.keys())

Del_motSVD=DAT['Del_motSVD']
Del_times=DAT['Del_times']
Del_trials=DAT['Del_trials']

#%% # resample the motion data to 10 ms resolution

# 10 ms resolution seem to be working well
num_new_times_imp = 210
resampled_motSVD = np.full((Del_motSVD.shape[0], Del_motSVD.shape[1], num_new_times_imp),np.nan)

for rep in range(Del_times.shape[0]):
    
    these_times = Del_times[rep,:]
    # 1. Define new time grid with 10ms (0.01s) intervals
    new_times = np.arange(np.min(these_times), np.max(these_times), 0.01)
    num_new_times = len(new_times)
    
    if num_new_times_imp != num_new_times:
        print('There is a mistake!')   
    for n in range(Del_motSVD.shape[0]):
        interp_func = interp1d(these_times, Del_motSVD[n, rep, :], kind='linear', fill_value="extrapolate")
        resampled_motSVD[n, rep, :] = interp_func(new_times) 

#%% set up the decoder

baseline_mean = np.mean(resampled_motSVD[:, :, :20], axis=2, keepdims=True)  # Shape: (100, 700, 1)
scaled_resampled_motSVD = resampled_motSVD - baseline_mean

n_rep = 50

# get spikes and labels in the right shape 
my_spikes, my_labels = shiftA_behav(scaled_resampled_motSVD,Del_trials,'all')

# set number of iterations
n_iter = 20
k_fold = 5
do_PCA = 0
n_output = int(my_spikes.shape[1]/k_fold)

predicted = np.zeros((n_iter,k_fold,n_output))
y_test = np.zeros((n_iter,k_fold,n_output))


for n in tqdm(range(n_iter)): 
    #select the neurons
    scores=my_spikes
    # run SVM
    n_comp = scores.shape[-1]
    predicted[n,:,:], y_test[n,:,:]= run_clf_kfold(n_comp, scores, my_labels,k_fold = k_fold, classifier ='SVM')

# let's save it! 

np.save(os.path.join(save_path,'PRED_Alldelays_Ashifted_original_animal_%d' %i + '_SVM.npy'),predicted)
np.save(os.path.join(save_path,'TEST_Alldelays_Ashifted_original_animal_%d' %i + '_SVM.npy'),y_test)

#%% and let's make it also with random labels

# shuffle the labels
shuffle_all_labels=np.array(random.sample(my_labels.tolist(),my_labels.shape[0])) # shuffle the labels here        
predicted_random = np.zeros((n_iter,k_fold,n_output))
y_test_random = np.zeros((n_iter,k_fold,n_output))

for n in tqdm(range(n_iter)): 
    #select the neurons
    scores=my_spikes
    # run SVM
    n_comp = scores.shape[-1]
    predicted_random[n,:,:], y_test_random[n,:,:]= run_clf_kfold(n_comp, scores, shuffle_all_labels,k_fold = k_fold, classifier ='SVM')

np.save(os.path.join(save_path,'PRED_Alldelays_Ashifted_original_random_animal_%d' %i + '_SVM.npy'),predicted_random)
np.save(os.path.join(save_path,'TEST_Alldelays_Ashifted_original_random_animal_%d' %i + '_SVM.npy'),y_test_random)

#%% and run the same but with only 1PC

predicted = np.zeros((n_iter,k_fold,n_output))
y_test = np.zeros((n_iter,k_fold,n_output))


for n in tqdm(range(n_iter)): 
    #select the neurons
    scores=my_spikes[:,:,:1]
    # run SVM
    n_comp = scores.shape[-1]
    predicted[n,:,:], y_test[n,:,:]= run_clf_kfold(n_comp, scores, my_labels,k_fold = k_fold, classifier ='SVM')

# let's save it! 

np.save(os.path.join(save_path,'PRED_Alldelays_Ashifted_1PC_original_animal_%d' %i + '_SVM.npy'),predicted)
np.save(os.path.join(save_path,'TEST_Alldelays_Ashifted_1PC_original_animal_%d' %i + '_SVM.npy'),y_test)

#%% and let's make it also with random labels

# shuffle the labels
shuffle_all_labels=np.array(random.sample(my_labels.tolist(),my_labels.shape[0])) # shuffle the labels here        
predicted_random = np.zeros((n_iter,k_fold,n_output))
y_test_random = np.zeros((n_iter,k_fold,n_output))

for n in tqdm(range(n_iter)): 
    #select the neurons
    scores=my_spikes[:,:,:1]
    # run SVM
    n_comp = scores.shape[-1]
    predicted_random[n,:,:], y_test_random[n,:,:]= run_clf_kfold(n_comp, scores, shuffle_all_labels,k_fold = k_fold, classifier ='SVM')

np.save(os.path.join(save_path,'PRED_Alldelays_Ashifted_1PC_original_random_animal_%d' %i + '_SVM.npy'),predicted_random)
np.save(os.path.join(save_path,'TEST_Alldelays_Ashifted_1PC_original_random_animal_%d' %i + '_SVM.npy'),y_test_random)
