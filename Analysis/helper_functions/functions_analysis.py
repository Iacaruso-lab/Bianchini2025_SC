# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 10:35:45 2022

@author: bianchg
"""

"""

This code has function useful for analysis and plotting 

"""

#%% import packages 

import os
import sys
import random
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="loaded more than 1 DLL from .libs:")

import numpy as np
import cv2
import pandas as pd
import scipy.io as mat
import scipy.stats as stats
import mat73
import imageio.v2 as imageio
from PIL import Image
import seaborn as sns
import squarify
import itertools
from itertools import combinations
from tqdm import tqdm
from math import log, sqrt
from scipy.stats import zscore

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from numpy.linalg import inv, solve
from numpy import matmul as mm

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.stats import ks_2samp, skew
from scipy.ndimage import gaussian_filter1d

import statsmodels.api as smi
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
from sklearn.metrics import r2_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Configure matplotlib
plt.rcParams["svg.fonttype"] = "none"


#%% logistic pca from here: https://github.com/brudfors/logistic-PCA-Tipping

def logistic_pca(X, num_components=None, num_iter=32):
    """Logistic principal component analysis (PCA).
    Parameters
    ----------
    X : (num_samples, num_dimensions) ndarray
        Data matrix.
    num_components : int, optional
        Number of PCA components.
    num_iter : int, default=32
        Number iterations for fitting model.
    Returns
    ----------
    W : (num_dimensions, num_components) ndarray
        Estimated projection matrix.
    mu : (num_components, num_samples) ndarray
        Estimated latent variables.
    b : (num_dimensions, 1) ndarray
        Estimated bias.
    Reference
    ----------
    Tipping, Michael E. "Probabilistic visualisation of high-dimensional binary data." 
    Advances in neural information processing systems (1999): 592-598.
    """
    num_samples = X.shape[0]
    num_dimensions = X.shape[1]
    num_components = _get_num_components(num_components, num_samples, num_dimensions)
    # shorthands
    N = num_samples
    D = num_dimensions
    K = num_components
    # initialise
    I = np.eye(K)
    W = np.random.randn(D, K)
    mu = np.random.randn(K, N)
    b = np.random.randn(D, 1)    
    C = np.repeat(I[:, :, np.newaxis], N, axis=2)
    xi = np.ones((N, D))  # the variational parameters
    # functions
    sig = lambda x: 1/(1 + np.exp(-x))
    lam = lambda x: (0.5 - sig(x))/(2*x)
    # fit model
    for iter in range(num_iter):
        # 1.obtain the sufficient statistics for the approximated posterior 
        # distribution of latent variables given each observation
        for n in range(N):
            # get sample
            x_n = X[n, :][:, None]
            # compute approximation
            lam_n = lam(xi[n, :])[:, None]
            # update
            C[:, :, n] = inv(I - 2*mm(W.T, lam_n*W))
            mu[:, n] = mm(C[:, :, n], mm(W.T, x_n - 0.5 + 2*lam_n*b))[:, 0]
        # 2.optimise the variational parameters in in order to make the 
        # approximation as close as possible
        for n in range(N):
            # posterior statistics
            z = mu[:, n][:, None]
            E_zz = C[:, :, n] + mm(z, z.T)
            # xi squared
            xixi = np.sum(W*mm(W, E_zz), axis=1, keepdims=True) \
                   + 2*b*mm(W, z) + b**2
            # update
            xi[n, :] = np.sqrt(np.abs(xixi[:, 0]))
        # 3.update model parameters
        E_zhzh = np.zeros((K + 1, K + 1, N))
        for n in range(N):
            z = mu[:, n][:, None]
            E_zhzh[:-1, :-1, n] = C[:, :, n] + mm(z, z.T)
            E_zhzh[:-1, -1, n] = z[:, 0]
            E_zhzh[-1, :-1, n] = z[:, 0]
            E_zhzh[-1, -1, n] = 1
        E_zh = np.append(mu, np.ones((1, N)), axis=0)
        for i in range(D):
            # compute approximation
            lam_i = lam(xi[:, i])[None][None]
            # gradient and Hessian
            H = np.sum(2*lam_i*E_zhzh, axis=2)
            g = mm(E_zh, X[:, i] - 0.5)
            # invert
            wh_i = -solve(H, g[:, None])
            wh_i = wh_i[:, 0]
            # update
            W[i, :] = wh_i[:K]
            b[i] = wh_i[K]
        
        X_reduced = np.dot(W.transpose(),X.transpose()).transpose()
        
    return W, mu, b, X_reduced


def normal_pca(X, num_components=None, zero_mean=True):
    """Principal component analysis (PCA).
    Parameters
    ----------
    X : (num_samples, num_dimensions) ndarray
        Data matrix.
    num_components : int, optional
        Number of PCA components.
    zero_mean : bool, default=True
        Zero mean data.
    Returns
    ----------
    W : (num_dimensions, num_components) ndarray
        Principal axes.        
    mu : (num_components, ) ndarray
        Principal components.    
    """
    num_samples = X.shape[0]
    num_dimensions = X.shape[1]
    num_components = _get_num_components(num_components, num_samples, num_dimensions)
    if zero_mean:        
        # zero mean
        X -= np.mean(X, axis=0)  
    # compute covariance matrix
    Xcov = np.cov(X, rowvar=False)
    # eigen decomposition
    mu, W = np.linalg.eig(Xcov)
    # sort descending order
    idx = np.argsort(mu)[::-1]
    W = W[:,idx]
    mu = mu[idx]
    # extract components
    mu = mu[:num_components]
    W = W[:, :num_components]
    
    X_reduced = np.dot(W.transpose(),X.transpose()).transpose()
    
    return W, mu, X_reduced
    
    
def _get_num_components(num_components, num_samples, num_dimensions):
    """Get number of components (clusters).
    """
    if num_components is None:
        num_components = min(num_samples, num_dimensions)    

    return num_components   

# code to create the sums for V+Ashifted trials for delay before and after protocol 

def shift_sum_BA(spikes_tot,sub_mean=1,factor =1):
    # Set the random seed for reproducibility
    np.random.seed(42)
    # general variables
    n_rep=50
    length_window = 25*factor
    stim_onset=98*factor # -10 ms
    stim_off=stim_onset+length_window # 250 ms
    
    #take the ones you already have
    existing_trials=spikes_tot[:,:,stim_onset:stim_off]

    # take vis and aud baseline and put in an array
    baseline=np.array([spikes_tot[:,-100:-50,:length_window],spikes_tot[:,-50:,:length_window]]) #take baseline of unisensory stimuli
    baseline = np.swapaxes(baseline,0,1)
    baseline = np.swapaxes(baseline,1,2)
    baseline = np.swapaxes(baseline,3,2)

    # get the V trial
    V = spikes_tot[:,-100:-50,stim_onset:stim_off] 
    V_temp = spikes_tot[:,-100:-50,:] 
    # create sum AV
    A = spikes_tot[:,-50:,stim_onset:stim_off]  
    A_temp = spikes_tot[:,-50:,:]  
    sum_AV = []
    if factor == 1:
        delays = [10*factor,5*factor,3*factor,0,3*factor,5*factor,10*factor]
    else:
        delays = [10*factor,5*factor,2.5*factor,0,2.5*factor,5*factor,10*factor]
    
    for count,tr in enumerate(delays):
        tr = int(tr)
        if tr == 0:
            uni = A
            shift = np.copy(V)
        elif count <=2:
            print(tr)
            
            # create the shiftV
            shift= V_temp[:,:,stim_onset-tr:stim_off-tr]  
            uni = np.copy(A)
        elif count >3:
            # create the shiftA
            shift= A_temp[:,:,stim_onset-tr:stim_off-tr]  
            uni = np.copy(V)

        # sum them together
        AV = shift + uni 
        
        if sub_mean ==1:
            final_baseline = []
            # and remove the baseline
            for i in range(AV.shape[1]):
                this_baseline = baseline[:,i,:,:].reshape(baseline.shape[0],-1)
                pos = np.array(random.sample(list(np.arange(this_baseline.shape[1])),AV.shape[2])).reshape(-1) 
                #final_baseline.append(np.nanmean(this_baseline, axis=-1, keepdims=True))
                final_baseline.append(this_baseline[:,pos[0]])
            final_baseline = np.array(final_baseline)
            final_baseline = np.swapaxes(final_baseline,0,1)   
            final_baseline = final_baseline.reshape(final_baseline.shape[0],final_baseline.shape[1],1)         
            # and remove the baseline 
            sum_AV.append(AV - final_baseline)
        else:
            sum_AV.append(AV)
    
    sum_AV = np.array(sum_AV)
    sum_AV = np.swapaxes(sum_AV,0,1)
    sum_AV=sum_AV.reshape(spikes_tot.shape[0],-1,length_window)
    #sum_AV = np.where(sum_AV < 0, 0, sum_AV) #to avoid having negative values

    # now append all the responses together
    all_spikes=np.concatenate((existing_trials,sum_AV),1)
    #all_spikes = all_spikes - np.mean(all_spikes,axis=2,keepdims=True)
    
    #create trial labels
    all_labels = np.repeat(np.arange(all_spikes.shape[1]/n_rep),n_rep)
    
    if all_labels.shape[0] == all_spikes.shape[1]:
        print('its correct')
    else:
        print('SOMETHING IS WRONG')
        
    return all_spikes, all_labels

## function to

def shift_sum_with_baseline(spikes_tot,trials,sub_mean=1):
    
    # general variables
    n_rep=50
    length_window = 30
    stim_onset=90 # -10 ms
    stim_off=stim_onset+length_window # 250 ms
    
    #blank trial
    no_stim=spikes_tot[:,-100:-50,:length_window] #take baseline of visual stim
    #take the ones you already have
    existing_trials=spikes_tot[:,:,stim_onset:stim_off]

    # take vis and aud baseline and put in an array
    baseline=np.array([spikes_tot[:,-100:-50,:length_window],spikes_tot[:,-50:,:length_window]]) #take baseline of unisensory stimuli
    baseline = np.swapaxes(baseline,0,1)
    baseline = np.swapaxes(baseline,1,2)
    baseline = np.swapaxes(baseline,3,2)

    # get the V trial
    V = spikes_tot[:,-100:-50,stim_onset:stim_off] 

    # create sum AV
    A = spikes_tot[:,-50:,:]  
    sum_AV = []
    for tr in range(len(trials)-2):
        step=tr
        # create the shiftA
        shiftA= A[:,:,stim_onset-step:stim_off-step]  

        # sum them together
        AV = shiftA + V 
        
        if sub_mean ==1:
            final_baseline = []
            # and remove the baseline
            for i in range(AV.shape[1]):
                this_baseline = baseline[:,i,:,:].reshape(baseline.shape[0],-1)
                pos = np.array(random.sample(list(np.arange(this_baseline.shape[1])),AV.shape[2])).reshape(-1) 
                #final_baseline.append(np.nanmean(this_baseline, axis=-1, keepdims=True))
                final_baseline.append(this_baseline[:,pos[0]])
            final_baseline = np.array(final_baseline)
            final_baseline = np.swapaxes(final_baseline,0,1)   
            final_baseline = final_baseline.reshape(final_baseline.shape[0],final_baseline.shape[1],1)         
            # and remove the baseline 
            sum_AV.append(AV - final_baseline)
        else:
            sum_AV.append(AV)
    
    sum_AV = np.array(sum_AV)
    sum_AV = np.swapaxes(sum_AV,0,1)
    sum_AV=sum_AV.reshape(spikes_tot.shape[0],-1,length_window)

    # now append all the responses together
    all_spikes=np.concatenate((existing_trials,sum_AV,no_stim),1)
    
    #create trial labels
    all_labels = np.repeat(np.arange(all_spikes.shape[1]/n_rep),n_rep)
    
    if all_labels.shape[0] == all_spikes.shape[1]:
        print('its correct')
    else:
        print('SOMETHING IS WRONG')
        
    return all_spikes, all_labels

#%% fit a 1d gaussian 

def gauss_offset(x,a,x0,sigma,b): #gaussian with offset term. Terms to fit are amplitude(a), mean(x0), sd(sigma) and offset (b)
    
    return a*np.sqrt(sigma)*np.exp(-(x-x0)**2/(2*sigma**2)) + b

#%% set up the initial parameters for gaussian fit 

def init_par(vals):
    amp = 1
    mean = np.argmax(vals)
    sigma = 1
    b = np.min(vals)
    p0 = [amp, mean,sigma, b] # initial parameters

    return p0      

#%% get the position of a specific neuron from all the neurons to the only responsive ones 

def pos_big_array(which_id,responsive):
    
    whos = np.arange(len(responsive))
    which_responsive = whos[responsive==1]
    
    pos_id = which_responsive[which_id]
    return pos_id

#%% plot neuron responses of random neurons, input how many is the number of random neurons you want to visualize

def plot_random_IFR(IFRs,raster,how_many,mod,choice,responsive,peaks):
    
    # make sure it's an array
    IFRs = np.array(IFRs)
    # find where are the neurons of choice
    choice_IDs = np.array((np.where(mod==choice))).reshape(-1).tolist()
    ids = random.sample(choice_IDs,how_many)    
    # plot them
    for i in range(how_many):
        actual = IFRs[ids[i],:,:] # now this is shaped trials x time
        maxFR = np.max(actual)
        raster_now = raster[ids[i]]
        plt.figure()
        for t in range(actual.shape[0]):
            ax = plt.subplot(int(np.ceil(actual.shape[0]/2)),2,t+1)
            x = raster_now['x'][t]
            y = raster_now['y'][t]
            ax.grid(False)
            ax.set_xlim([-0.3,0.7])
            ax.set_ylim([0,53])
            start = 0
            hight = 50
            ax.scatter(x, y,edgecolor='gainsboro',s=10,c='gainsboro')
            ax.plot([start, start+0.1], [hight,hight],linewidth = 10,c = 'gold',alpha=0.2 ,zorder =1)  #ax.axvline(x=0.1, color='k', linestyle='--') 
            ax.set_xticks([])
            ax.set_yticks([])  
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.axis('off')
            if t==0:
                ax.set_xticks([0,0.4])
                ax.set_yticks([0,50])
                myFont = 'Verdana'
                mySize = 10
                ax.set_ylabel('Trials', fontname=myFont, fontsize=mySize)
                ax.set_xlabel('time (ms)', fontname=myFont, fontsize=mySize)
            
            # twin object for two different y-axis on the sample plot
            ax2 = ax.twinx().twiny()#ax2,ax3=ax.twinx(), ax.twiny()
            ax2.plot(actual[t,:],c='black')
            ax2.get_xaxis().set_visible(False)
            ax2.get_yaxis().set_visible(False)
            ax2.grid(False)
            ax2.set_xlim([0,1000])
            ax2.set_ylim([0,maxFR+10])  
            ax2.spines["top"].set_visible(False)
            ax2.spines["bottom"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            ax2.spines["left"].set_visible(False)
            ax2.axis('off')
            if t==0:
                ax2.set_yticks([0,np.round(maxFR,-1)])  
                ax2.set_ylabel('FR (Hz)', fontname=myFont, fontsize=mySize)
            else:
                ax2.set_yticks([])
            plt.title('trial = %d' %t)
            
        plt.suptitle('neuron ID = %d ' %ids[i])
        pos_id = pos_big_array(ids[i],responsive)

        plot_delay_tuning(pos_id[0],peaks)
        
#%% plot specific neurons but only the stimuli you want

def plot_single_IFR(IFRs,raster,id_neuron):
    
    # make sure it's an array
    IFRs = np.array(IFRs) 
    
    actual = IFRs[id_neuron,:,:] # now this is shaped trials x time
    maxFR = np.max(actual)
    raster_now = raster[id_neuron]
    plt.figure()
    
    for t in range(actual.shape[0]):
        ax = plt.subplot(int(np.ceil(actual.shape[0]/2)),2,t+1)
        x = raster_now['x'][t]
        y = raster_now['y'][t]
        ax.grid(False)
        ax.set_xlim([-0.3,0.7])
        ax.set_ylim([0,53])
        start = 0
        hight = 50
        ax.scatter(x, y,edgecolor='gainsboro',s=10,c='gainsboro')
        ax.plot([start, start+0.1], [hight,hight],linewidth = 10,c = 'gold',alpha=0.2 ,zorder =1)  #ax.axvline(x=0.1, color='k', linestyle='--') 
        ax.set_xticks([])
        ax.set_yticks([])  
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.axis('off')
        if t==0:
            ax.set_xticks([0,0.4])
            ax.set_yticks([0,50])
            myFont = 'Verdana'
            mySize = 10
            ax.set_ylabel('Trials', fontname=myFont, fontsize=mySize)
            ax.set_xlabel('time (ms)', fontname=myFont, fontsize=mySize)
        
        # twin object for two different y-axis on the sample plot
        ax2 = ax.twinx().twiny()#ax2,ax3=ax.twinx(), ax.twiny()
        ax2.plot(actual[t,:],c='black')
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        ax2.grid(False)
        ax2.set_xlim([0,1000])
        ax2.set_ylim([0,maxFR+10])  
        ax2.spines["top"].set_visible(False)
        ax2.spines["bottom"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.axis('off')
        if t==0:
            ax2.set_yticks([0,np.round(maxFR,-1)])  
            ax2.set_ylabel('FR (Hz)', fontname=myFont, fontsize=mySize)
        else:
            ax2.set_yticks([])
        plt.title('trial = %d' %t)
       
    plt.suptitle('neuron ID = %d ' %id_neuron)

#%% plot specific neurons but only the stimuli you want

def plot_single_IFR_only(IFRs,id_neuron):
    
    # make sure it's an array
    IFRs = np.array(IFRs) 
    
    actual = IFRs[id_neuron,:,:] # now this is shaped trials x time
    maxFR = np.max(actual)
    trial_types = ['Del 0', 'Del 10', 'Del 20', 'Del 30', 'Del 40', 'Del 50', 'Del 60', 'Del 70', 'Del 80', 'Del 90', 'Del 100', 'Vis', 'Aud']
    plt.subplots(figsize=(8, 4),tight_layout=True)

    
    for t in range(actual.shape[0]):
        ax = plt.subplot(2,int(np.ceil(actual.shape[0]/2)),t+1)        
        # twin object for two different y-axis on the sample plot
        vertical_line_x = 300
        ax.axvline(x=vertical_line_x, color='red', linestyle='--')
        ax.plot(actual[t,:],c='black')
        ax.get_xaxis().set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        if t!=0:
            ax.get_yaxis().set_visible(False)
            ax.grid(False)
            ax.set_xlim([0,1000])
            ax.set_ylim([0,maxFR+10])  
            
            ax.spines["bottom"].set_visible(False)            
            ax.spines["left"].set_visible(False)
            ax.axis('off')
        myFont = 'Verdana'
        mySize = 10
        if t==0:
            ax.set_yticks([0,np.round(maxFR,-1)])  
            ax.set_ylabel('FR (Hz)', fontname=myFont, fontsize=mySize)
        else:
            ax.set_yticks([])
        plt.title(trial_types[t])
       
    plt.suptitle('neuron ID = %d ' %id_neuron)

#%% plot specific neurons but only the stimuli you want

def plot_ids_IFR(IFRs,raster,ids,n_stim,delay_cond):
    
    # make sure it's an array
    IFRs = np.array(IFRs) 
    
    pos = np.arange(len(ids)*n_stim)
    pos = pos.reshape(n_stim,-1)
    
    # plot them
    plt.figure()
    for i in range(len(ids)):
        
        actual = IFRs[ids[i],:,:] # now this is shaped trials x time
        maxFR = np.max(actual)
        raster_now = raster[ids[i]]
        stims = [actual.shape[0]-2, actual.shape[0]-1, int(delay_cond[i])]
        
        for t in range(n_stim):
            ax = plt.subplot(n_stim,len(ids),pos[t,i]+1)
            x = raster_now['x'][stims[t]]
            y = raster_now['y'][stims[t]]
            ax.grid(False)
            ax.set_xlim([-0.3,0.5])
            ax.set_ylim([0,53])
            start = 0
            hight = 50
            ax.scatter(x, y,edgecolor='gainsboro',s=10,c='gainsboro')
            ax.plot([start, start+0.1], [hight,hight],linewidth = 10,c = 'gold',alpha=0.2 ,zorder =1)  #ax.axvline(x=0.1, color='k', linestyle='--') 
            ax.set_xticks([])
            ax.set_yticks([])  
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.axis('off')
            if t==0:
                ax.set_xticks([0,0.4])
                ax.set_yticks([0,50])
                myFont = 'Verdana'
                mySize = 10
                ax.set_ylabel('Trials', fontname=myFont, fontsize=mySize)
                ax.set_xlabel('time (ms)', fontname=myFont, fontsize=mySize)
            
            # twin object for two different y-axis on the sample plot
            ax2 = ax.twinx().twiny()#ax2,ax3=ax.twinx(), ax.twiny()
            ax2.plot(actual[stims[t],:],c='black')
            ax2.get_xaxis().set_visible(False)
            ax2.get_yaxis().set_visible(False)
            ax2.grid(False)
            ax2.set_xlim([0,800])
            ax2.set_ylim([0,maxFR+10])  
            ax2.spines["top"].set_visible(False)
            ax2.spines["bottom"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            ax2.spines["left"].set_visible(False)
            ax2.axis('off')
            if t==0:
                ax2.set_yticks([0,np.round(maxFR,-1)])  
                ax2.set_ylabel('FR (Hz)', fontname=myFont, fontsize=mySize)
            else:
                ax2.set_yticks([])
            if t ==0:
                plt.title('neuron = %d' %i)

#%% plot specific delays for multiple neurons 

def plot_ids_delays(IFRs,raster,ids,n_stim,delays):
    
    # make sure it's an array
    IFRs = np.array(IFRs) 
    
    pos = np.arange(len(ids)*n_stim)
    pos = pos.reshape(n_stim,-1)
    
    stims = delays
    # plot them
    plt.figure()
    for i in range(len(ids)):
        
        actual = IFRs[ids[i],:,:] # now this is shaped trials x time
        maxFR = np.max(actual)
        raster_now = raster[ids[i]]
        
        
        for t in range(n_stim):
            ax = plt.subplot(n_stim,len(ids),pos[t,i]+1)
            x = raster_now['x'][stims[t]]
            y = raster_now['y'][stims[t]]
            ax.grid(False)
            ax.set_xlim([-0.3,0.5])
            ax.set_ylim([0,53])
            start = 0 + (stims[t]*0.01)
            if stims[t] == 12 or stims[t] == 11:
                start = 0
            hight = 50
            ax.scatter(x, y,edgecolor='gainsboro',s=10,c='gainsboro')
            ax.plot([start, start+0.1], [hight,hight],linewidth = 10,c = 'gold',alpha=0.2 ,zorder =1)  #ax.axvline(x=0.1, color='k', linestyle='--') 
            ax.set_xticks([])
            ax.set_yticks([])  
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.axis('off')
            if t==0:
                ax.set_xticks([0,0.4])
                ax.set_yticks([0,50])
                myFont = 'Verdana'
                mySize = 10
                ax.set_ylabel('Trials', fontname=myFont, fontsize=mySize)
                ax.set_xlabel('time (ms)', fontname=myFont, fontsize=mySize)
            
            # twin object for two different y-axis on the sample plot
            ax2 = ax.twinx().twiny()#ax2,ax3=ax.twinx(), ax.twiny()
            ax2.plot(actual[stims[t],:],c='black')
            ax2.get_xaxis().set_visible(False)
            ax2.get_yaxis().set_visible(False)
            ax2.grid(False)
            ax2.set_xlim([0,800])
            ax2.set_ylim([0,maxFR+10])  
            ax2.spines["top"].set_visible(False)
            ax2.spines["bottom"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            ax2.spines["left"].set_visible(False)
            ax2.axis('off')
            if t==0:
                ax2.set_yticks([0,np.round(maxFR,-1)])  
                ax2.set_ylabel('FR (Hz)', fontname=myFont, fontsize=mySize)
            else:
                ax2.set_yticks([])
            if i ==0:
                plt.title('delay tr = %d' %stims[t])
                
#%% plot tuning curve for delays

def plot_delay_tuning(ids,peaks):
    
    curr_peaks = peaks[ids,:].reshape(-1)
    
    plt.figure()
    plt.plot(curr_peaks[:-2],c='k')      
    plt.hlines(y=curr_peaks[-2], xmin=0,xmax=curr_peaks[:-2].shape[0], color='b', linestyle='--') # vis FR
    plt.hlines(y=curr_peaks[-1], xmin=0,xmax=curr_peaks[:-2].shape[0], color='r', linestyle='--') # aud FR
    x_positions = np.arange(curr_peaks[:-2].shape[0])
    x_labels = [0,10,20,30,40,50,60,70,80,90,100]
    plt.xticks(x_positions, x_labels)  # Set text labels and properties.
    plt.ylabel('FR (Hz)')
    plt.xlabel('Delay (ms)')
    plt.title('Tuning curve')

#%% plot tuning curve for delays

def plot_delay_tuning_norm(ids,peakFR,norm=0,color='k'):
    
    curr_peaks = np.squeeze(peakFR[ids,:,:])
    
    meanFR = np.mean(curr_peaks,axis=1)
    max_uni = np.max(curr_peaks[-2:])
    max_multi = np.max(curr_peaks[:-2])
    if norm ==1:
        curr_peaks = (curr_peaks-max_uni)/(max_multi-max_uni)
    

    plt.plot(curr_peaks[:-2],c=color)      
    x_positions = np.arange(curr_peaks[:-2].shape[0])
    x_labels = [0,10,20,30,40,50,60,70,80,90,100]
    plt.xticks(x_positions, x_labels)  # Set text labels and properties.
    plt.ylabel('FR (Hz)')
    plt.xlabel('Delay (ms)')
    plt.title('Tuning curve')
    
#%% plot mean scores for SVM output 

def plot_scoresCLF(scores,color,axis=plt):

    score_final=np.nanmean(scores,axis=0)
    score_final_low=np.percentile(scores,25,axis=0)
    score_final_high=np.percentile(scores,75,axis=0)
    axis.plot(score_final,c=color)
    axis.fill_between(np.arange(len(score_final)),score_final_low,score_final_high,color=color,alpha=0.2)
    max_val = np.nanmax(score_final_high)

    return max_val

#%% plot mean scores for SVM output 

def scatter_scoresCLF(loc,scores,color,axis=plt):

    score_final=np.mean(scores,axis=0)
    score_final_low=np.percentile(scores,25,axis=0)
    score_final_high=np.percentile(scores,75,axis=0)
    axis.plot(loc,score_final,c=color)
    axis.fill_between(loc,score_final_low,score_final_high,color=color,alpha=0.2)

    max_val = np.nanmax(score_final_high)

    return max_val

#%% plot mean scores for SVM output  and also display values on top

def plot_scoresCLF_and_values(scores,color,vals,type_text):

    score_final=np.mean(scores,axis=0)
    score_final_low=np.percentile(scores,25,axis=0)
    score_final_high=np.percentile(scores,75,axis=0)
    plt.plot(score_final,c=color)
    plt.fill_between(np.arange(len(score_final)),score_final_low,score_final_high,color=color,alpha=0.2)
    
    x = np.arange(len(score_final))
    y = np.ceil(score_final_high)
    fixed_y = max(y) + 2
    
    if type_text == 'experiment':
        info = (vals[:,3]).astype(int)
    elif type_text == 'animal':
        info = (vals[:,2]).astype(int)
    elif type_text == 'neurons':
        info = (vals[:,1]).astype(int)
    elif type_text == 'length':
        info = (vals[:,0]).astype(int)
        
    for i, j in zip(x, y):
        plt.text(i, fixed_y, str(info[i]), ha='center', fontsize=10)

#%% Define function with *args syntax
def kruskal_test(*args):
    # Check each input array for NaNs and exclude it if found
    args_clean = []
    for arr in args:
        if not np.isnan(arr).any():
            args_clean.append(arr)            
    
    stat, p_value = stats.kruskal(*args_clean)
    
    return stat, p_value
#%% create empty variable with nans

def nans(shape, dtype=float):
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a

#%% shift auditory rsponse for movement control

def shiftA_behav(spikes,trials,cond,resp = 1,mod = 1,x = 1):

    # general variables
    n_rep=50
    length_window = 110
    stim_onset=40 # 
    stim_off=stim_onset+length_window

    #blank trial
    no_stim=spikes[:,-50:,stim_onset:stim_off] #no stim condition
    #take the ones you already have
    existing_trials=spikes[:,:-50,stim_onset:stim_off]
    
    # create shift A trials - the A trial is always the last one in the matrix
    A = spikes[:,-100:-50,:]
    shiftA = []    
    for tr in range(1,len(trials)-3):
        step=tr
        A_new = A[:,:,stim_onset-step:stim_off-step]   
        shiftA.append(A_new) 
        
    shiftA=np.array(shiftA) 
    shiftA = np.swapaxes(shiftA,0,1)
    shiftA=shiftA.reshape(spikes.shape[0],-1,length_window)
    
    # now append all the responses together
    all_spikes=np.concatenate((existing_trials,shiftA,no_stim),1)
    
    #create trial labels
    all_labels = np.repeat(np.arange(all_spikes.shape[1]/n_rep),n_rep)
    
    if all_labels.shape[0] == all_spikes.shape[1]:
        print('its correct')
    else:
        print('SOMETHING IS WRONG')
        print(all_labels.shape[0])
        print(all_spikes.shape[1])
        print(existing_trials.shape)
        print(shiftA.shape)
        print(no_stim.shape)
        
    return all_spikes, all_labels

#%% sum AV 

def shift_sum(spikes_tot,trials,sub_mean=1):
    
    # general variables
    n_rep=50
    length_window = 25
    stim_onset=98 # -10 ms
    stim_off=stim_onset+length_window # 250 ms
    
    #blank trial
    no_stim=spikes_tot[:,-100:-50,:length_window] #take baseline of visual stim
    #take the ones you already have
    existing_trials=spikes_tot[:,:,stim_onset:stim_off]

    # take vis and aud baseline and put in an array
    baseline=np.array([spikes_tot[:,-100:-50,:length_window],spikes_tot[:,-50:,:length_window]]) #take baseline of unisensory stimuli
    baseline = np.swapaxes(baseline,0,1)
    baseline = np.swapaxes(baseline,1,2)
    baseline = np.swapaxes(baseline,3,2)

    # get the V trial
    V = spikes_tot[:,-100:-50,stim_onset:stim_off] 

    # create sum AV
    A = spikes_tot[:,-50:,:]  
    sum_AV = []
    for tr in range(len(trials)-2):
        step=tr
        # create the shiftA
        shiftA= A[:,:,stim_onset-step:stim_off-step]  

        # sum them together
        AV = shiftA + V 
        
        if sub_mean ==1:
            final_baseline = []
            # and remove the baseline
            for i in range(AV.shape[1]):
                this_baseline = baseline[:,i,:,:].reshape(baseline.shape[0],-1)
                pos = np.array(random.sample(list(np.arange(this_baseline.shape[1])),AV.shape[2])).reshape(-1) 
                #final_baseline.append(np.nanmean(this_baseline, axis=-1, keepdims=True))
                final_baseline.append(this_baseline[:,pos[0]])
            final_baseline = np.array(final_baseline)
            final_baseline = np.swapaxes(final_baseline,0,1)   
            final_baseline = final_baseline.reshape(final_baseline.shape[0],final_baseline.shape[1],1)         
            # and remove the baseline 
            sum_AV.append(AV - final_baseline)
        else:
            sum_AV.append(AV)
    
    sum_AV = np.array(sum_AV)
    sum_AV = np.swapaxes(sum_AV,0,1)
   
    sum_AV=sum_AV.reshape(spikes_tot.shape[0],-1,length_window)
    #sum_AV = np.where(sum_AV < 0, 0, sum_AV) #to avoid having negative values

    # now append all the responses together
    all_spikes=np.concatenate((existing_trials,sum_AV,no_stim),1)
    #all_spikes = all_spikes - np.mean(all_spikes,axis=2,keepdims=True)
    
    #create trial labels
    all_labels = np.repeat(np.arange(all_spikes.shape[1]/n_rep),n_rep)
    
    if all_labels.shape[0] == all_spikes.shape[1]:
        print('its correct')
    else:
        print('SOMETHING IS WRONG')
        
    return all_spikes, all_labels

#%%
def shift_sum_1ms(spikes_tot,trials,sub_mean=1):
    
    # general variables
    n_rep=50
    length_window = 25*10
    stim_onset=98*10 # -10 ms
    stim_off=stim_onset+length_window # 250 ms
    
    #blank trial
    no_stim=spikes_tot[:,-100:-50,:length_window] #take baseline of visual stim
    #take the ones you already have
    existing_trials=spikes_tot[:,:,stim_onset:stim_off]

    # take vis and aud baseline and put in an array
    baseline=np.array([spikes_tot[:,-100:-50,:length_window],spikes_tot[:,-50:,:length_window]]) #take baseline of unisensory stimuli
    baseline = np.swapaxes(baseline,0,1)
    baseline = np.swapaxes(baseline,1,2)
    baseline = np.swapaxes(baseline,3,2)

    # get the V trial
    V = spikes_tot[:,-100:-50,stim_onset:stim_off] 

    # create sum AV
    A = spikes_tot[:,-50:,:]  
    sum_AV = []
    for tr in range(len(trials)-2):
        step=tr
        # create the shiftA
        shiftA= A[:,:,stim_onset-step:stim_off-step]  

        # sum them together
        AV = shiftA + V 
        
        if sub_mean ==1:
            final_baseline = []
            # and remove the baseline
            for i in range(AV.shape[1]):
                this_baseline = baseline[:,i,:,:].reshape(baseline.shape[0],-1)
                pos = np.array(random.sample(list(np.arange(this_baseline.shape[1])),AV.shape[2])).reshape(-1) 
                #final_baseline.append(np.nanmean(this_baseline, axis=-1, keepdims=True))
                final_baseline.append(this_baseline[:,pos[0]])
            final_baseline = np.array(final_baseline)
            final_baseline = np.swapaxes(final_baseline,0,1)   
            final_baseline = final_baseline.reshape(final_baseline.shape[0],final_baseline.shape[1],1)         
            # and remove the baseline 
            sum_AV.append(AV - final_baseline)
        else:
            sum_AV.append(AV)
    
    sum_AV = np.array(sum_AV)
    sum_AV = np.swapaxes(sum_AV,0,1)
    sum_AV=sum_AV.reshape(spikes_tot.shape[0],-1,length_window)
    #sum_AV = np.where(sum_AV < 0, 0, sum_AV) #to avoid having negative values

    # now append all the responses together
    all_spikes=np.concatenate((existing_trials,sum_AV,no_stim),1)
    #all_spikes = all_spikes - np.mean(all_spikes,axis=2,keepdims=True)
    
    #create trial labels
    all_labels = np.repeat(np.arange(all_spikes.shape[1]/n_rep),n_rep)
    
    if all_labels.shape[0] == all_spikes.shape[1]:
        print('its correct')
    else:
        print('SOMETHING IS WRONG')
        
    return all_spikes, all_labels

#%%
def shift_sum_5ms(spikes_tot,trials,sub_mean=1):
    
    # general variables
    n_rep=50
    length_window = 25*2
    stim_onset=98*2 # -10 ms
    stim_off=stim_onset+length_window # 250 ms
    
    #blank trial
    no_stim=spikes_tot[:,-100:-50,:length_window] #take baseline of visual stim
    #take the ones you already have
    existing_trials=spikes_tot[:,:,stim_onset:stim_off]

    # take vis and aud baseline and put in an array
    baseline=np.array([spikes_tot[:,-100:-50,:length_window],spikes_tot[:,-50:,:length_window]]) #take baseline of unisensory stimuli
    baseline = np.swapaxes(baseline,0,1)
    baseline = np.swapaxes(baseline,1,2)
    baseline = np.swapaxes(baseline,3,2)

    # get the V trial
    V = spikes_tot[:,-100:-50,stim_onset:stim_off] 

    # create sum AV
    A = spikes_tot[:,-50:,:]  
    sum_AV = []
    for tr in range(len(trials)-2):
        step=tr
        # create the shiftA
        shiftA= A[:,:,stim_onset-step:stim_off-step]  

        # sum them together
        AV = shiftA + V 
        
        if sub_mean ==1:
            final_baseline = []
            # and remove the baseline
            for i in range(AV.shape[1]):
                this_baseline = baseline[:,i,:,:].reshape(baseline.shape[0],-1)
                pos = np.array(random.sample(list(np.arange(this_baseline.shape[1])),AV.shape[2])).reshape(-1) 
                #final_baseline.append(np.nanmean(this_baseline, axis=-1, keepdims=True))
                final_baseline.append(this_baseline[:,pos[0]])
            final_baseline = np.array(final_baseline)
            final_baseline = np.swapaxes(final_baseline,0,1)   
            final_baseline = final_baseline.reshape(final_baseline.shape[0],final_baseline.shape[1],1)         
            # and remove the baseline 
            sum_AV.append(AV - final_baseline)
        else:
            sum_AV.append(AV)
    
    sum_AV = np.array(sum_AV)
    sum_AV = np.swapaxes(sum_AV,0,1)
    sum_AV=sum_AV.reshape(spikes_tot.shape[0],-1,length_window)
    #sum_AV = np.where(sum_AV < 0, 0, sum_AV) #to avoid having negative values

    # now append all the responses together
    all_spikes=np.concatenate((existing_trials,sum_AV,no_stim),1)
    #all_spikes = all_spikes - np.mean(all_spikes,axis=2,keepdims=True)
    
    #create trial labels
    all_labels = np.repeat(np.arange(all_spikes.shape[1]/n_rep),n_rep)
    
    if all_labels.shape[0] == all_spikes.shape[1]:
        print('its correct')
    else:
        print('SOMETHING IS WRONG')
        
    return all_spikes, all_labels
#%% shift A trials

def shiftA(spikes_tot,trials,cond,resp = 1,mod = 1,x = 1):
#    spikes_tot = spikes_tot[a[300],:,:]
    # get only the neurons you are interested in
    if cond == 'all':
        spikes = spikes_tot
    elif cond == 'resp':
        spikes = spikes_tot[resp==1,:,:]
    elif cond == 'sub':
        spikes = spikes_tot[mod==x,:,:]
    
    # general variables
    n_rep=50
    length_window = 25
    stim_onset=98 # -10 ms
    stim_off=stim_onset+length_window # 250 ms
    
    #blank trial
    no_stim=spikes[:,-100:-50,:length_window] #take baseline of visual stim
    #take the ones you already have
    existing_trials=spikes[:,:,stim_onset:stim_off]
    
    # create shift A trials - the A trial is always the last one in the matrix
    A = spikes[:,-50:,:]
    shiftA = []    
    for tr in range(1,len(trials)-2):
        step=tr
        A_new = A[:,:,stim_onset-step:stim_off-step]   
        shiftA.append(A_new) 
        
    shiftA=np.array(shiftA) 
    shiftA = np.swapaxes(shiftA,0,1)
    #shiftA = np.swapaxes(shiftA,1,2)
    shiftA=shiftA.reshape(spikes.shape[0],-1,length_window)
    
    # now append all the responses together
    all_spikes=np.concatenate((existing_trials,shiftA,no_stim),1)
    #all_spikes = all_spikes - np.mean(all_spikes,axis=2,keepdims=True)
    
    #create trial labels
    all_labels = np.repeat(np.arange(all_spikes.shape[1]/n_rep),n_rep)
    
    if all_labels.shape[0] == all_spikes.shape[1]:
        print('its correct')
    else:
        print('SOMETHING IS WRONG')
        
    return all_spikes, all_labels

#%% shift A trials

def shiftA_20ms(spikes_tot,trials,cond,resp = 1,mod = 1,x = 1):
#    spikes_tot = spikes_tot[a[300],:,:]
    # get only the neurons you are interested in
    if cond == 'all':
        spikes = spikes_tot
    elif cond == 'resp':
        spikes = spikes_tot[resp==1,:,:]
    elif cond == 'sub':
        spikes = spikes_tot[mod==x,:,:]
    
    # general variables
    n_rep=50
    length_window = int(25/2)
    stim_onset=int(98/2) # -10 ms
    stim_off=stim_onset+length_window # 250 ms
    
    #blank trial
    no_stim=spikes[:,-100:-50,:length_window] #take baseline of visual stim
    #take the ones you already have
    existing_trials=spikes[:,:,stim_onset:stim_off]
    
    # create shift A trials - the A trial is always the last one in the matrix
    A = spikes[:,-50:,:]
    shiftA = []    
    for tr in range(1,len(trials)-2):
        step=tr
        A_new = A[:,:,stim_onset-step:stim_off-step]   
        shiftA.append(A_new) 
        
    shiftA=np.array(shiftA) 
    shiftA = np.swapaxes(shiftA,0,1)
    #shiftA = np.swapaxes(shiftA,1,2)
    shiftA=shiftA.reshape(spikes.shape[0],-1,length_window)
    
    # now append all the responses together
    all_spikes=np.concatenate((existing_trials,shiftA,no_stim),1)
    #all_spikes = all_spikes - np.mean(all_spikes,axis=2,keepdims=True)
    
    #create trial labels
    all_labels = np.repeat(np.arange(all_spikes.shape[1]/n_rep),n_rep)
    
    if all_labels.shape[0] == all_spikes.shape[1]:
        print('its correct')
    else:
        print('SOMETHING IS WRONG')
        
    return all_spikes, all_labels

#%% shift A trials

def shiftA_10ms(spikes_tot,trials,cond,resp = 1,mod = 1,x = 1):
#    spikes_tot = spikes_tot[a[300],:,:]
    # get only the neurons you are interested in
    if cond == 'all':
        spikes = spikes_tot
    elif cond == 'resp':
        spikes = spikes_tot[resp==1,:,:]
    elif cond == 'sub':
        spikes = spikes_tot[mod==x,:,:]
    
    # general variables
    n_rep=50
    length_window = 25
    stim_onset=98 # -10 ms
    stim_off=stim_onset+length_window # 250 ms
    
    #blank trial
    no_stim=spikes[:,-100:-50,:length_window] #take baseline of visual stim
    #take the ones you already have
    existing_trials=spikes[:,:,stim_onset:stim_off]
    
    # create shift A trials - the A trial is always the last one in the matrix
    A = spikes[:,-50:,:]
    shiftA = []    
    for tr in range(1,len(trials)-2):
        step=tr
        A_new = A[:,:,stim_onset-step:stim_off-step]   
        shiftA.append(A_new) 
        
    shiftA=np.array(shiftA) 
    shiftA = np.swapaxes(shiftA,0,1)
    #shiftA = np.swapaxes(shiftA,1,2)
    shiftA=shiftA.reshape(spikes.shape[0],-1,length_window)
    
    # now append all the responses together
    all_spikes=np.concatenate((existing_trials,shiftA,no_stim),1)
    #all_spikes = all_spikes - np.mean(all_spikes,axis=2,keepdims=True)
    
    #create trial labels
    all_labels = np.repeat(np.arange(all_spikes.shape[1]/n_rep),n_rep)
    
    if all_labels.shape[0] == all_spikes.shape[1]:
        print('its correct')
    else:
        print('SOMETHING IS WRONG')
        
    return all_spikes, all_labels

#%% shift A trials

def shiftA_5ms(spikes_tot,trials,cond,resp = 1,mod = 1,x = 1):
#    spikes_tot = spikes_tot[a[300],:,:]
    # get only the neurons you are interested in
    if cond == 'all':
        spikes = spikes_tot
    elif cond == 'resp':
        spikes = spikes_tot[resp==1,:,:]
    elif cond == 'sub':
        spikes = spikes_tot[mod==x,:,:]
    
    # general variables
    n_rep=50
    length_window = 25*2
    stim_onset=98*2 # -10 ms
    stim_off=stim_onset+length_window # 250 ms
    
    #blank trial
    no_stim=spikes[:,-100:-50,:length_window] #take baseline of visual stim
    #take the ones you already have
    existing_trials=spikes[:,:,stim_onset:stim_off]
    
    # create shift A trials - the A trial is always the last one in the matrix
    A = spikes[:,-50:,:]
    shiftA = []    
    for tr in range(1,len(trials)-2):
        step=tr*2
        A_new = A[:,:,stim_onset-step:stim_off-step]   
        shiftA.append(A_new) 
        
    shiftA=np.array(shiftA) 
    shiftA = np.swapaxes(shiftA,0,1)
    #shiftA = np.swapaxes(shiftA,1,2)
    shiftA=shiftA.reshape(spikes.shape[0],-1,length_window)
    
    # now append all the responses together
    all_spikes=np.concatenate((existing_trials,shiftA,no_stim),1)
    #all_spikes = all_spikes - np.mean(all_spikes,axis=2,keepdims=True)
    
    #create trial labels
    all_labels = np.repeat(np.arange(all_spikes.shape[1]/n_rep),n_rep)
    
    if all_labels.shape[0] == all_spikes.shape[1]:
        print('its correct')
    else:
        print('SOMETHING IS WRONG')
        
    return all_spikes, all_labels

#%% shift A trials

def shiftA_1ms(spikes_tot,trials,cond,resp = 1,mod = 1,x = 1):
#    spikes_tot = spikes_tot[a[300],:,:]
    # get only the neurons you are interested in
    if cond == 'all':
        spikes = spikes_tot
    elif cond == 'resp':
        spikes = spikes_tot[resp==1,:,:]
    elif cond == 'sub':
        spikes = spikes_tot[mod==x,:,:]
    
    # general variables
    n_rep=50
    length_window = 25*10
    stim_onset=98*10 # -10 ms
    stim_off=stim_onset+length_window # 250 ms
    
    #blank trial
    no_stim=spikes[:,-100:-50,:length_window] #take baseline of visual stim
    #take the ones you already have
    existing_trials=spikes[:,:,stim_onset:stim_off]
    
    # create shift A trials - the A trial is always the last one in the matrix
    A = spikes[:,-50:,:]
    shiftA = []    
    for tr in range(1,len(trials)-2):
        step=tr*10
        A_new = A[:,:,stim_onset-step:stim_off-step]   
        shiftA.append(A_new) 
        
    shiftA=np.array(shiftA) 
    shiftA = np.swapaxes(shiftA,0,1)
    #shiftA = np.swapaxes(shiftA,1,2)
    shiftA=shiftA.reshape(spikes.shape[0],-1,length_window)
    
    # now append all the responses together
    all_spikes=np.concatenate((existing_trials,shiftA,no_stim),1)
    #all_spikes = all_spikes - np.mean(all_spikes,axis=2,keepdims=True)
    
    #create trial labels
    all_labels = np.repeat(np.arange(all_spikes.shape[1]/n_rep),n_rep)
    
    if all_labels.shape[0] == all_spikes.shape[1]:
        print('its correct')
    else:
        print('SOMETHING IS WRONG')
        
    return all_spikes, all_labels

#%% function to run PCA for SVM

def doPCA_forSVM(all_spikes):
    
    # get the time dimensione
    n_comp = all_spikes.shape[-1]
    # reshape the spikes
    spikes_reshape = np.reshape(all_spikes,(all_spikes.shape[0]*all_spikes.shape[1],all_spikes.shape[2])) #N1 all repets x time
    
    if spikes_reshape.shape[1] == n_comp:
        
        # do PCA
        pca = PCA(n_components=n_comp)
        scores=pca.fit_transform(spikes_reshape) #PCs
        
        #reshape the PCs values
        scores_reshape=np.reshape(scores,(all_spikes.shape[0],all_spikes.shape[1],n_comp))
        
    else:
        print('SOMETHING IS WRONG')
        
    return scores_reshape

#%% function to run SVM 

def runSVM(n_comp, scores, labels):
    
    # how many features to take
    features = scores[:,:,:n_comp]
    # yopu need to reshape it for the SVM - x axis is trials
    features = np.swapaxes(features,0,1)
    features = features.reshape(features.shape[0],-1)
    
    # split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(features,labels, test_size=0.2)
    # zscore the PCs
    scaler = StandardScaler()

    X_train_Z = scaler.fit_transform(X_train)
    X_test_Z = scaler.transform(X_test)
    # run SVM
    svm = LinearSVC(max_iter=10000)
    svm.fit(X_train_Z, y_train)
    # pretict on test data
    predicted = svm.predict(X_test_Z)
    SVM_scores= svm.score(X_test_Z,y_test) 
    
    return predicted, y_test, SVM_scores

#%% function to run randomforest classifier

def runRFC(n_comp, scores, labels):
    
    # how many features to take
    features = scores[:,:,:n_comp]
    # yopu need to reshape it for the SVM - x axis is trials
    features = np.swapaxes(features,0,1)
    features = features.reshape(features.shape[0],-1)
    
    # split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(features,labels, test_size=0.2)
    # zscore the PCs
    scaler = StandardScaler()

    X_train_Z = scaler.fit_transform(X_train)
    X_test_Z = scaler.transform(X_test)
    # run random forest classifier
    
    clf = RandomForestClassifier()
    
    clf.fit(X_train_Z, y_train)
    # pretict on test data
    predicted = clf.predict(X_test_Z)
    RFC_scores= clf.score(X_test_Z,y_test) 
    
    return predicted, y_test, RFC_scores

#%% function to select and run classifier

def run_clf(n_comp, scores, labels,classifier ='SVM'):
    
    # how many features to take
    features = scores[:,:,:n_comp]
    # yopu need to reshape it for the SVM - x axis is trials
    features = np.swapaxes(features,0,1)
    features = features.reshape(features.shape[0],-1)
    
    # split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(features,labels, test_size=0.2)
    # zscore the PCs
    scaler = StandardScaler()

    X_train_Z = scaler.fit_transform(X_train)
    X_test_Z = scaler.transform(X_test)
    
    if classifier == 'SVM':
        #run SVM
        clf = LinearSVC(max_iter=10000)
    elif classifier  == 'RFC':
        # run random forest classifier    
        clf = RandomForestClassifier()
    elif classifier == 'DTC':
        clf = DecisionTreeClassifier(random_state=0)
    
    # then just fit it
    clf.fit(X_train_Z, y_train)
    # pretict on test data
    predicted = clf.predict(X_test_Z)
    RFC_scores= clf.score(X_test_Z,y_test) 
    
    return predicted, y_test, RFC_scores

#%% function to select and run classifier

def run_clf_kfold(n_comp, scores, labels,k_fold = 10, classifier ='SVM'):
    
    # how many features to take
    features = scores[:,:,:n_comp]
    # yopu need to reshape it for the SVM - x axis is trials
    features = np.swapaxes(features,0,1)
    features = features.reshape(features.shape[0],-1)

    # define kfold
    kf = KFold(n_splits=k_fold,shuffle=True,random_state=42)
    
    #initiate empty arrays
    predicted = []
    test = []
    # get the train and test index
    for train_index, test_index in kf.split(features):
    
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # zscore the PCs

        scaler = StandardScaler().fit(X_train)
        X_train_Z = scaler.transform(X_train)
        X_test_Z = scaler.transform(X_test)

        if classifier == 'SVM':
            #run SVM
            clf = SVC(kernel='linear')#, C=1.0, random_state=0, tol=1e-5)
            #clf = LinearSVC(dual='True', random_state=0, tol=1e-5)
          
        elif classifier  == 'RFC':
            # run random forest classifier    
            clf = RandomForestClassifier()
        elif classifier == 'DTC':
            clf = DecisionTreeClassifier(random_state=0)
    
        # then just fit it
        clf.fit(X_train_Z, y_train)
        # pretict on test data
        predicted.append(clf.predict(X_test_Z))
        test.append(y_test)

    P = np.array(predicted)
    T = np.array(test)

    return P,T

#%% function to select and run classifier

def run_clf_kfold_1feature(n_comp, scores, labels,k_fold = 10, classifier ='SVM'):
    
    # how many features to take
    features = scores
    # yopu need to reshape it for the SVM - x axis is trials
    features = np.swapaxes(features,0,1)
    features = features.reshape(features.shape[0],-1)

    # define kfold
    kf = KFold(n_splits=k_fold,shuffle=True,random_state=0)
    
    #initiate empty arrays
    predicted = []
    test = []
    # get the train and test index
    for train_index, test_index in kf.split(features):
    
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # zscore the PCs
        scaler = StandardScaler()

        X_train_Z = scaler.fit_transform(X_train)
        X_test_Z = scaler.transform(X_test)
    
        if classifier == 'SVM':
            #run SVM
            clf = LinearSVC()
        elif classifier  == 'RFC':
            # run random forest classifier    
            clf = RandomForestClassifier()
        elif classifier == 'DTC':
            clf = DecisionTreeClassifier(random_state=0)
    
        # then just fit it
        clf.fit(X_train_Z, y_train)
        # pretict on test data
        predicted.append(clf.predict(X_test_Z))
        test.append(y_test)

    P = np.array(predicted)
    T = np.array(test)

    return P,T
#%% function to select and run classifier for single neurons

def run_clf_1n(n_comp, scores, labels,classifier ='SVM'):
    
    # how many features to take
    features = scores[:,:n_comp]
    # yopu need to reshape it for the SVM - x axis is trials
    #features = np.swapaxes(features,0,1)
    features = features.reshape(features.shape[0],-1)
    
    # split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(features,labels, test_size=0.2)
    # zscore the PCs
    scaler = StandardScaler()

    X_train_Z = scaler.fit_transform(X_train)
    X_test_Z = scaler.transform(X_test)
    
    if classifier == 'SVM':
        #run SVM
        clf = LinearSVC(max_iter=10000)
    elif classifier  == 'RFC':
        # run random forest classifier    
        clf = RandomForestClassifier()
    elif classifier == 'DTC':
        clf = DecisionTreeClassifier(random_state=0)
    
    # then just fit it
    clf.fit(X_train_Z, y_train)
    # pretict on test data
    predicted = clf.predict(X_test_Z)
    RFC_scores= clf.score(X_test_Z,y_test) 
    
    return predicted, y_test, RFC_scores

#%% function to run randomforest classifier

def runDTC(n_comp, scores, labels):
    
    # how many features to take
    features = scores[:,:,:n_comp]
    # yopu need to reshape it for the SVM - x axis is trials
    features = np.swapaxes(features,0,1)
    features = features.reshape(features.shape[0],-1)
    
    # split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(features,labels, test_size=0.2)
    # zscore the PCs
    scaler = StandardScaler()

    X_train_Z = scaler.fit_transform(X_train)
    X_test_Z = scaler.transform(X_test)
    # run decision tree classifier
    clf = DecisionTreeClassifier(random_state=0)
    
    clf.fit(X_train_Z, y_train)
    # pretict on test data
    predicted = clf.predict(X_test_Z)
    DTC_scores= clf.score(X_test_Z,y_test) 
    
    return predicted, y_test, DTC_scores

#%% function to normalise the data between range 0 to 1

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

#%% Define a linear function to fit the data

def linear_func(x, a, b):
    return a * x + b

#%% Define exponential function to fit the data

import scipy.optimize

# Define your exponential function
def exponential(x, y_0, y_1, lamb):
    return y_0 * np.exp(x * -lamb) + y_1

# Define your cost function
def cost_function(params, x, y):
    return np.sum(np.abs(y - exponential(x, *params)))

#%% function to plot histogram in percentage

def plot_hist_percentage(bin_edges,data):

    # Compute the histogram
    hist, bins = np.histogram(data, bins=bin_edges)

    # Compute the percentage values
    percentage = (hist / len(data)) * 100

    return percentage, bins

#%% makeSpatialBinnedMap

def makeSpatialBinnedMap(spatialBin = 100, factor=10):
        
        #um_per_pixel = 9 #this is from measuring the WF FOV size
         
        #spatialBin_px = int(spatialBin/um_per_pixel)
                    
                       
        #nY,nX = ref.shape[0:2]

        nBins_x = factor#np.ceil(nX/spatialBin_px)
        nBins_y = factor#np.ceil(nY/spatialBin_px)
                        
        X,Y = np.meshgrid(np.arange(0,nBins_x), np.arange(0,nBins_y))
            
        Z = X*100 + Y
            
        matrix = np.repeat(Z, spatialBin, axis=1).repeat(spatialBin, axis=0)            
        matrix = matrix[0:nBins_y,0:nBins_x]
                
        return matrix
    
#%% makeMeanValue_bySpatialBin

def makeMeanValue_bySpatialBin(df, binned_map,x_lab,y_lab,lab,factor):
    # df = azi_peak_red_vis_sig
      
    x = np.ceil(np.array(df[x_lab])*factor)
    y = np.ceil(np.array(df[y_lab])*factor)
    x=x.astype(int)
    y=y.astype(int)
    
    val=np.array(df[lab])

    binIndices = binned_map[y,x]            
    binIndices_unique = np.unique(binIndices)
    #name = df.keys()[0]
    final_binIndices_unique=[]
   
    for b in binIndices_unique:
        ids=np.where(binIndices==b)
        if len(ids[0])>=10:
            final_binIndices_unique.append(b)        
        
    meanVal = []
    final_binIndices_unique = np.array(final_binIndices_unique)
    for b in final_binIndices_unique:
        ids=np.where(final_binIndices_unique==b)
        meanVal.append(np.nanmean(val[ids]))
           
    binned_mean_map = binned_map.copy()
    allBins = np.unique(binned_map)
    
    for i in range(len(allBins)): 
        
        if allBins[i] in final_binIndices_unique:
            whichBin = np.nonzero(final_binIndices_unique == allBins[i])[0]
            binned_mean_map[binned_mean_map == allBins[i]] = meanVal[int(whichBin)]
        else:
            binned_mean_map[binned_mean_map == allBins[i]] = np.nan
         
    return binned_mean_map.tolist()


#%% makePlot_bySpatialBin, plot binned map for anything

# def makeMeanValue_bySpatialBin(df,spatialBin,x_lab,y_lab,lab,factor):
    
#     binned_map = makeSpatialBinnedMap(spatialBin = spatialBin,factor=factor)
#     binned_map=binned_map.astype(np.float32)
#     binned_values_map = makeMeanValue_bySpatialBin(df, binned_map,x_lab,y_lab,lab,factor)
    
#     return binned_mean_map
#     #fig=plt.figure()
#     #plt.imshow(binned_values_map)
#     #plt.colorbar()
#     #plt.xlabel(x_lab)
#     #plt.ylabel(y_lab)
#     #plt.title('MAP for ' + lab)        
    
#     #return binned_values_map.tolist()

#%% makeBins_SC

def makeBins_SC(coord,n_bins):
    
    hist, edges = np.histogram(coord, bins=n_bins)
    id=np.digitize(coord,edges)
    id[id==n_bins+1]=n_bins
    
    return id,edges

#%% getBinIDs

def getBinIDs(n_var, var1, var2, n_bins, var3=[]):
    """
    Returns the bin ID for each data point given the values of its variables, as well as the bin groups and their lengths.

    Parameters:
    n_var (int): Number of variables.
    var1 (numpy array): Values of the first variable.
    var2 (numpy array): Values of the second variable.
    n_bins (int): Number of bins.
    var3 (numpy array, optional): Values of the third variable. Default is an empty array.

    Returns:
    groups (numpy array): Bin groups, where each row corresponds to a bin and each column corresponds to a variable.
    lengths (list of int): Lengths of the bin groups.
    ID_neurons (numpy array): Bin ID for each data point.
    bins (int): Total number of bins.

    """

    if n_var == 2: # If there are two variables
        id = np.zeros((len(var1), n_var)) # Create a 2D array of zeros with the same number of rows as the input arrays
        id[:, n_var-2] = var1[:] # Fill the first column with the values of the first variable
        id[:, n_var-1] = var2[:] # Fill the second column with the values of the second variable
        
        n_comb = n_bins ** n_var # Calculate the total number of bin combinations
        
        seq = np.arange(id.shape[0]) # Create an array of integers from 0 to the number of rows in id
        
        groupsA = [] # Create an empty list for the first variable groups
        groupsB = [] # Create an empty list for the second variable groups
        for i in range(n_bins):
            temp_ids1 = seq[id[:, 0] == (i+1)] # Get the indices of the rows where the first variable is equal to i+1
            groupsA.append(np.repeat(i+1, n_bins)) # Add n_bins copies of i+1 to the first variable groups list    
            for i2 in range(n_bins):  
                temp_ids2 = seq[id[:, 1] == (i2+1)] # Get the indices of the rows where the second variable is equal to i2+1
                temp_ids = np.intersect1d(temp_ids1, temp_ids2) # Find the intersection between the two sets of indices
                groupsB.append(i2+1) # Add i2+1 to the second variable groups list
                
        groups = np.zeros((n_comb, 2)) # Create a 2D array of zeros with the number of rows equal to the number of bin combinations and the number of columns equal to the number of variables
        groups[:, 0] = np.reshape(np.array(groupsA), -1) # Fill the first column with the first variable groups
        groups[:, 1] = np.array(groupsB) # Fill the second column with the second variable groups
        bins = len(groups) # Get the total number of bins
        lengths = [] # Create an empty list for the bin group lengths
        ID_neurons = np.zeros(seq.shape) # Create an array of zeros with the same shape as seq for storing the bin ID for each data point
        for g in range(len(groups)): # For each bin group
            gx = groups[g] # Get the current bin group
            temp_ids1 = seq[id[:, 0] == gx[0]] # Get the indices of the rows where the first variable is equal to the first value of the current bin
            temp_ids2=seq[id[:,1]==gx[1]]
            temp_ids=np.intersect1d(temp_ids1, temp_ids2)
            lengths.append(temp_ids.shape[0])
            ID_neurons[temp_ids]=g+1
            
    elif n_var==3:
        
        id=np.zeros((len(var1),n_var))
        id[:,n_var-3]=var1
        id[:,n_var-2]=var2
        id[:,n_var-1]=var3
        
        n_comb = n_bins ** n_var
        
        seq=np.arange(id.shape[0])
    
        groupsA=[]
        groupsB=[]
        groupsC=[]
        for i in range(n_bins):
            temp_ids1 = seq[id[:,0] == (i+1)]
            if n_bins==2:
                groupsA.append(np.repeat(i+1,4))   
            else:
                groupsA.append(np.repeat(i+1,n_bins*n_var))    
            for i2 in range(n_bins):  
                temp_ids2=seq[id[:,1] == (i2+1)]                
                groupsB.append(np.repeat(i2+1,n_bins))
                for i3 in range(n_bins):  
                    temp_ids3=seq[id[:,2] == (i3+1)]
                    temp_ids=np.intersect1d(temp_ids1, temp_ids2)
                    temp_ids=np.intersect1d(temp_ids, temp_ids3)
                    groupsC.append(i3+1)
        
        groups=np.zeros((n_comb,n_var))
        groups[:,0]=np.reshape(np.array(groupsA),-1)
        groups[:,1]=np.reshape(np.array(groupsB),-1)
        groups[:,2]=np.array(groupsC)
        bins = len(groups)
        lengths=[]
        ID_neurons=np.zeros(seq.shape)
        for g in range(bins):
            gx=groups[g]
            temp_ids1=seq[id[:,0]==gx[0]]
            temp_ids2=seq[id[:,1]==gx[1]]
            temp_ids3=seq[id[:,2]==gx[2]]
            temp_ids=np.intersect1d(temp_ids1, temp_ids2)
            temp_ids=np.intersect1d(temp_ids, temp_ids3)
            lengths.append(temp_ids.shape[0])
            ID_neurons[temp_ids]=g+1
    
    return groups,lengths,ID_neurons,bins
#%%

def getBinIDs_v2(n_var, *vars, n_bins):
    """
    Returns the bin ID for each data point given the values of its variables, as well as the bin groups and their lengths.

    Parameters:
    n_var (int): Number of variables.
    *vars: Values of the variables.
    n_bins (int): Number of bins.

    Returns:
    groups (numpy array): Bin groups, where each row corresponds to a bin and each column corresponds to a variable.
    lengths (list of int): Lengths of the bin groups.
    ID_neurons (numpy array): Bin ID for each data point.
    bins (int): Total number of bins.

    """
    vars = list(vars)
    id = np.zeros((len(vars[0]), n_var))
    for i in range(n_var):
        id[:, n_var-i-1] = vars[i][:]

    n_comb = n_bins ** n_var
    seq = np.arange(id.shape[0])

    groups = [[] for _ in range(n_var)]
    for i in range(n_var):
        for j in range(n_bins):
            temp_ids = seq[id[:, i] == (j + 1)]
            groups[i].append(np.repeat(j+1, len(temp_ids)))

    groups = np.zeros((n_comb, n_var))
    for i in range(n_var):
        groups[:, i] = np.reshape(np.array(groups[i]), -1)

    bins = len(groups)
    lengths = []
    ID_neurons = np.zeros(seq.shape)

    for g in range(bins):
        gx = groups[g]
        temp_ids = seq
        for i in range(n_var):
            temp_ids = temp_ids[id[:, i] == gx[i]]
        lengths.append(temp_ids.shape[0])
        ID_neurons[temp_ids] = g + 1

    return groups, lengths, ID_neurons, bins


#%%

def myPlotSettings(fig,ax,ytitle,xtitle,title):
    myFont = 'Verdana'
    mySize = 10
    plt.rcParams["font.family"] = myFont
    plt.rcParams["font.size"] = mySize

    ax.set_ylabel(ytitle, fontname=myFont, fontsize=mySize)
    ax.set_xlabel(xtitle, fontname=myFont, fontsize=mySize)
    ax.set_title(title, fontname=myFont, fontsize=mySize, weight = 'bold')
    for tick in ax.get_xticklabels():
        tick.set_fontname(myFont)
        tick.set_fontsize(mySize)        
    for tick in ax.get_yticklabels():
        tick.set_fontname(myFont)
        tick.set_fontsize(mySize)    
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)
    right = ax.spines["right"]
    right.set_visible(False)
    top = ax.spines["top"]
    top.set_visible(False) 
    


#%% LINEAR REGRESSION

# X = np.transpose(np.array([np.array(df['ML']),np.array(df['AP']),np.array(df['depth']),np.array(df['peaktime_vis']),np.array(df['peaktime_aud'])])) #all parameters
# Y =  np.transpose(np.array([np.array(df['pref_delay'])]))#parameter to predict

# # TRAINING DATA
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# # FIT MODEL
# reg = LinearRegression().fit(X_train,y_train)
# vals_cross[i] = np.mean(cross_val_score(reg,X_train,y_train,cv=KFold(5)))

# # TEST MODEL
# y_pred, y_std = gpr.predict(X_test, return_std=True) # predicted values
# vals_test[i] = gpr.score(X_test, y_test)


# vals=np.zeros((comp_final.shape[1],1))

# Y=comp_final[:,i]

# score=reg.score(X, Y)
# vals[i]=score


# # FIT MODEL
# gpr.fit(X_train, y_train)
# vals_cross[i] = np.mean(cross_val_score(gpr,X_train,y_train,cv=KFold(5)))

# # TEST MODEL
# y_pred, y_std = gpr.predict(X_test, return_std=True) # predicted values
# vals_test[i] = gpr.score(X_test, y_test)

#%%

def xicor(X, Y, ties=True):
    np.random.seed(42)
    n = len(X)
    r = np.argsort(X)
    if ties:
        l = np.array([np.sum(Y >= np.sort(Y[r[i]])) for i in range(n)])
        r_copy = np.copy(r)
        for j in range(n):
            if np.sum(r_copy[j] == r_copy) > 1:
                tie_index = np.where(r_copy == r_copy[j])[0]
                r_copy[tie_index] = np.random.choice(
                    r_copy[tie_index] - np.arange(0, np.sum(r_copy[j] == r_copy)), 
                    size=np.sum(tie_index), 
                    replace=False
                )
        return 1 - n * np.sum(np.abs(r_copy[1:] - r_copy[:n-1])) / (2 * np.sum(l * (n - l)))
    else:
        r = np.array([np.sum(Y >= np.sort(Y[r[i]])) for i in range(n)])
        return 1 - 3 * np.sum(np.abs(r[1:] - r[:n-1])) / (n**2 - 1)

#%% MoranI - by Alex EW

def getMoranI(data, weightMatrix, permutations=999):
    N = len(data)
    W = np.sum(weightMatrix)
    xMean = np.nanmean(data)
    xVar = np.nanvar(data)
    meanSub = data - xMean
    
    observed_I = np.nansum((weightMatrix * meanSub).T * meanSub) / (xVar * W)
    
    permutations_I = []
    for _ in range(permutations):
        shuffled_data = np.random.permutation(data)
        shuffled_meanSub = shuffled_data - np.nanmean(shuffled_data)
        permuted_I = np.nansum((weightMatrix * shuffled_meanSub).T * shuffled_meanSub) / (xVar * W)
        permutations_I.append(permuted_I)
    
    p_value = (np.sum(np.array(permutations_I) >= observed_I) + 1) / (permutations + 1)
    
    return observed_I, p_value

#%% spatial lag - by Alex EW

def getSpatialLag(data,weightMatrix):
     '''
     spatial lag is the mean of 'data' for all rois close to the target roi,
     difference is the mean of the difference between 'data' for each roi and 'data' for the rois around it
     '''
     N = len(data)        
             
     spatialLag, difference = [],[]
     for roi in range(N):
         # print(str(roi))
                       
         idx = np.nonzero(weightMatrix[roi,:] > 0.5)[0]
                     
         if len(idx) > 0:                    
             diff = np.nanmean(abs(data[roi] - data[idx]))
             lag = np.nanmean(data[idx]-np.nanmean(data))                 
         else:
             diff = np.nan
             lag = np.nan
                     
         difference.append(diff)
         spatialLag.append(lag)
             
     spatialLag = np.array(spatialLag)
     difference = np.array(difference)
                                         
     return spatialLag, difference

# %%
