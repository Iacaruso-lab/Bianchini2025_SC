#%% import packages

import mat73
import numpy as np
import sys
from tqdm import tqdm
import random
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
import seaborn as sns

sys.path.insert(0, 'Z:\\home\\shared\\Gaia\\Coliseum\\Delays\\paper_code\\Analysis\\helper_functions')
from functions_analysis import shiftA_behav,doPCA_forSVM, run_clf_kfold

#%% load data
data_path = 'Z:\\home\\shared\\Gaia\\Coliseum\\Delays\\paper_code\\Datasets\\movement_control_datasets\\raw_data'
videoFolders = [os.path.join(dataPath, d) for d in os.listdir(dataPath) if os.path.isdir(os.path.join(dataPath, d))]  

save_path = 'Z:\\home\\shared\\Gaia\\Coliseum\\Delays\\paper_code\\Datasets\\movement_control_datasets\\decoder_analysis'

#%% load the dataset
 
i=int(list(sys.argv)[1])
print(i)

file=os.path.join(videoFolders[i],'dec_data.mat')
data_dict = mat73.loadmat(file)
DAT=data_dict['dec_data']

#check keys available
print(DAT.keys())

Vol_motSVD=DAT['Vol_motSVD']
Vol_times=DAT['Vol_times']

#%% # resample the motion data to 10 ms resolution

# 10 ms resolution seem to be working well
num_new_times_imp = 210
resampled_motSVD = np.full((Vol_motSVD.shape[0], Vol_motSVD.shape[1], num_new_times_imp),np.nan)

for rep in range(Vol_times.shape[0]):
    
    these_times = Vol_times[rep,:]
    # 1. Define new time grid with 10ms (0.01s) intervals
    new_times = np.arange(np.min(these_times), np.max(these_times), 0.01)
    num_new_times = len(new_times)
    
    if num_new_times_imp != num_new_times:
        print('There is a mistake!')   
    for n in range(Vol_motSVD.shape[0]):
        interp_func = interp1d(these_times, Vol_motSVD[n, rep, :], kind='linear', fill_value="extrapolate")
        resampled_motSVD[n, rep, :] = interp_func(new_times) 

#%% set up the decoder

n_rep = 50

#create trial labels
all_labels = np.repeat(np.arange(DAT['Vol_trials'].shape[0]),n_rep)

# let's try only yes sound no sound
# #window=[-0.5 1.6]
# new window from =10ms to 1s
noSound_data = resampled_motSVD[:,:50,40:-60]
Sound80dB = resampled_motSVD[:,-100:-50,40:-60]
Sound60dB = resampled_motSVD[:,200:250,40:-60]

#%% run decoder with no sound and sound 80 

new_resampled_motSVD = np.concatenate((Sound80dB,noSound_data),1)
new_resampled_motSVD.shape

my_spikes = new_resampled_motSVD
my_labels = all_labels[:100]

# set number of iterations
n_iter = 20
k_fold = 5
do_PCA = 0
n_output = int(my_spikes.shape[1]/k_fold)

predicted = np.zeros((n_iter,k_fold,n_output))
y_test = np.zeros((n_iter,k_fold,n_output))


for n in tqdm(range(n_iter)): 
    #select the neurons
    scores=my_spikes[:,:,:]
    # run SVM
    n_comp = scores.shape[-1]
    predicted[n,:,:], y_test[n,:,:]= run_clf_kfold(n_comp, scores, my_labels,k_fold = k_fold, classifier ='SVM')

# let's save it! 

np.save(os.path.join(save_path,'PRED_SoundNoSound_animal_%d' %i + '_SVM.npy'),predicted)
np.save(os.path.join(save_path,'TEST_SoundNoSound_animal_%d' %i + '_SVM.npy'),y_test)

# and let's make it also with random labels

# shuffle the labels
shuffle_all_labels=np.array(random.sample(my_labels.tolist(),my_labels.shape[0])) # shuffle the labels here        
predicted = np.zeros((n_iter,k_fold,n_output))
y_test = np.zeros((n_iter,k_fold,n_output))

for n in tqdm(range(n_iter)): 
    #select the neurons
    scores=my_spikes[:,:,:]
    # run SVM
    n_comp = scores.shape[-1]
    predicted[n,:,:], y_test[n,:,:]= run_clf_kfold(n_comp, scores, shuffle_all_labels,k_fold = k_fold, classifier ='SVM')

# let's save it! 

np.save(os.path.join(save_path,'PRED_SoundNoSound_random_animal_%d' %i + '_SVM.npy'),predicted)
np.save(os.path.join(save_path,'TEST_SoundNoSound_random_animal_%d' %i + '_SVM.npy'),y_test)


#%% run decoder with no sound and sound 60 

new_resampled_motSVD = np.concatenate((Sound60dB,noSound_data),1)
new_resampled_motSVD.shape

my_spikes = new_resampled_motSVD
my_labels = all_labels[:100]

# set number of iterations
n_iter = 20
k_fold = 5
do_PCA = 0
n_output = int(my_spikes.shape[1]/k_fold)

predicted = np.zeros((n_iter,k_fold,n_output))
y_test = np.zeros((n_iter,k_fold,n_output))


for n in tqdm(range(n_iter)): 
    #select the neurons
    scores=my_spikes[:,:,:]
    # run SVM
    n_comp = scores.shape[-1]
    predicted[n,:,:], y_test[n,:,:]= run_clf_kfold(n_comp, scores, my_labels,k_fold = k_fold, classifier ='SVM')

# let's save it! 

np.save(os.path.join(save_path,'PRED_SoundNoSound60dB_animal_%d' %i + '_SVM.npy'),predicted)
np.save(os.path.join(save_path,'TEST_SoundNoSound60dB_animal_%d' %i + '_SVM.npy'),y_test)

# and let's make it also with random labels

# shuffle the labels
shuffle_all_labels=np.array(random.sample(my_labels.tolist(),my_labels.shape[0])) # shuffle the labels here        
predicted = np.zeros((n_iter,k_fold,n_output))
y_test = np.zeros((n_iter,k_fold,n_output))

for n in tqdm(range(n_iter)): 
    #select the neurons
    scores=my_spikes[:,:,:]
    # run SVM
    n_comp = scores.shape[-1]
    predicted[n,:,:], y_test[n,:,:]= run_clf_kfold(n_comp, scores, shuffle_all_labels,k_fold = k_fold, classifier ='SVM')

# let's save it! 

np.save(os.path.join(save_path,'PRED_SoundNoSound60dB_random_animal_%d' %i + '_SVM.npy'),predicted)
np.save(os.path.join(save_path,'TEST_SoundNoSound60dB_random_animal_%d' %i + '_SVM.npy'),y_test)
