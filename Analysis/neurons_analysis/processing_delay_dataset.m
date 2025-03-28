%% load the data

close all
clear all
clc

on_cluster = 1;
which_dataset = 'delay_tuning_preprocessing'; % let's select the dataset
if on_cluster == 1
    file_path = fullfile('/camp/lab/iacarusof/home/shared/Gaia/Coliseum/Delays/paper_code/Datasets/neurons_datasets/pre_processing_datasets', [which_dataset '.mat']);
    addpath(genpath('/camp/lab/iacarusof/home/shared/Gaia/Coliseum/Delays/spikes-master')) % for converting to Phy
    addpath(genpath('/camp/lab/iacarusof/home/shared/Gaia/Coliseum/Delays/ZETA')) % for zeta testing
    addpath(genpath('/camp/lab/iacarusof/home/shared/Gaia/Coliseum/Delays/paper_code/Analysis/helper_functions')) % helper functions
else
    file_path = fullfile('Z:\home\shared\Gaia\Coliseum\Delays\paper_code\Datasets\neurons_datasets\pre_processing_datasets', [which_dataset '.mat']);
end

load(file_path)

%% Calculate IFR

for d=1:length(complete_dataset)
   
    s = complete_dataset(d).spikeTimes;
    e = complete_dataset(d).eventTimes -0.25;
    db=0.5;
    intSmoothSd=1;
    dt_ms = 1;
    line=linspace(0,db+0.01,1/(dt_ms*10^-3));
    % get the trials - % different datasets have different number of trials but the delyas 0-100 + V + A are in these trials
    which_trials = getWhichTrials(e);

    smooth_FR=zeros(length(which_trials),1000);
    for i=1:length(which_trials)
        events = e(:,1,which_trials(i));
        [~,~,sIFR] = getIFR(s,events,db,intSmoothSd);
        
        %correct for duplicates
        a = 0;
        b = 0.001;
        r = (b-a).*rand(10000,1) + a;
        [v, w] = unique( sIFR.vecTime, 'stable' );
        duplicate_indices = setdiff( 1:numel(sIFR.vecTime), w );
        sIFR.vecTime(duplicate_indices)=sIFR.vecTime(duplicate_indices)+r(duplicate_indices);

        %perform interpolation
        interp_result=interp1(sIFR.vecTime,sIFR.vecRate,line,'linear');
        
        smooth_FR(i,:) = movmean(interp_result,10);
    end
    neuron_IFR(d).mean=smooth_FR;
    d
end

% Save the IFR separately
file_path2 = fullfile('/camp/lab/iacarusof/home/shared/Gaia/Coliseum/Delays/paper_code/Datasets/neurons_datasets', ['psth_delay_tuning_dataset.mat']);
save(file_path2,'neuron_IFR','-v7.3')


%% Let's set up the ZETA testing - to test for each condition if there is a response

for d=1:length(complete_dataset)

    s = complete_dataset(d).spikeTimes;
    e = complete_dataset(d).eventTimes -0.25; 
    db = 0.5;
    intSmoothSd=0.5; %1
    
    n_trials = size(e,3);

    test = [];
    for i=1:n_trials
        events = e(:,1,i);
        [dblZetaP] = getZeta(s,events,db);
        test=[test;dblZetaP];
    end

    complete_dataset(d).Z_pval=test;
    d

end

%% Correct for multiple comparisons and save modality for each neuron

% First you need to correct for multiple comparison
for d = 1:length(complete_dataset)
    
    p_vals = complete_dataset(d).Z_pval;
    n_trials = length(p_vals);
    
    % different datasets have different number of trials but the delyas 0-100 + V + A are in these trials
    if n_trials == 16 
        which_trials = [1:11, 15, 16];
    elseif n_trials == 19
        which_trials = [1:11, 18, 19];
    elseif n_trials == 14
        which_trials = [1:13];
    end

    p_vals = p_vals(which_trials);
    complete_dataset(d).sig_vis = complete_dataset(d).output_boot(1,:);
    complete_dataset(d).sig_aud = complete_dataset(d).output_boot(2,:);

    % check max IFR if > 10
    IFR = neuron_IFR(d).mean;
    maxf=max(max(IFR(:,:)));

    % first correct for all multiple comparison
    p_val_corr = pval_adjust(p_vals,'BH');

    m = sum(find(p_val_corr(1:11)<0.05));
    complete_dataset(d).modality = 0;

    if any(p_val_corr<0.05) && maxf>10
        complete_dataset(d).responsive = 1;

        % get their modality
        if p_val_corr(12)<0.05 && p_val_corr(13)>0.05

            complete_dataset(d).modality = 1; %visual

        elseif p_val_corr(13)<0.05 && p_val_corr(12)>0.05
            complete_dataset(d).modality = 2; %auditory

        elseif p_val_corr(13)<0.05 && p_val_corr(12)<0.05
            complete_dataset(d).modality = 3; %aud-vis

        elseif p_val_corr(13)>0.05 && p_val_corr(12)>0.05 && m>0
            complete_dataset(d).modality = 4; %multi
        end
    else
        complete_dataset(d).responsive = 0;
        complete_dataset(d).modality = 0; %non responsive
    end

end

%% After determining responsive neurons run bootstrap analysis on responsive neurons 

% example code for this analysis can be found at - example_bootstrap_delays

%% Get the peak FR around a 10ms window 

%all the parameters
tr_n = 11;
db=0.25; % duration
smooth=0.5; % smoothing factor
dt_ms = 1;
line=linspace(0,db+0.01,1/(dt_ms*10^-3));

for d = 1:length(complete_dataset)

    s = complete_dataset(d).spikeTimes;
    e = complete_dataset(d).eventTimes-0.01;
    
    % get the trials
    which_trials = getWhichTrials(e);
    peak_FR = zeros(length(which_trials),1);

    for t = 1:length(which_trials)
        tr = which_trials(t);

        tr_trials = e(:,1,tr); %pool together trials        
        [~,~,sIFR]=getIFR(s,tr_trials,db,smooth);

        % I need to interpolate them because I need the time to be
        % the same between the two
        a = 0;
        b = 0.001;
        r = (b-a).*rand(10000,1) + a;

        % if there are duplicates in time adjust for it
        [~, w] = unique( sIFR.vecTime, 'stable' );
        duplicate_indices = setdiff( 1:numel(sIFR.vecTime), w );
        sIFR.vecTime(duplicate_indices)=sIFR.vecTime(duplicate_indices)+r(duplicate_indices);

        %perform interpolation
        tr=interp1(sIFR.vecTime,sIFR.vecRate,line,'linear');

        % find the max value for each
        [~,pos]=max(tr);
        if pos<6
            max_tr=mean(tr(1:pos+5));
        elseif pos>6
            max_tr=mean(tr(pos-5:pos+5));
        end
        peak_FR(t) = max_tr;

    end
    complete_dataset(d).peakFR = peak_FR;  
d
end

%% Extract the raster

%1. get the spikes in zeros and ones and the raster to save
for d = 1:length(neuron_matrix)
    
    s = neuron_matrix(d).spikeTimes;
    e = neuron_matrix(d).eventTimes;
    binSize=0.001; %this determines how big the bins are - you can vary this from 0.0001 to whatever fits best
    window=[-0.3 0.3];

    which_trials = getWhichTrials(e);

    tot_spikes=[];
    for tr=1:length(which_trials)
        events = e(:,1,which_trials(tr));
        [~, ~, rasterX, rasterY, ~, ba] = psthAndBA(s, events, window, binSize);        

        % save the raster
        raster(d).tr(tr).x=rasterX;
        raster(d).tr(tr).y=rasterY;
    end

    d
end

% save the raster already
save('/camp/lab/iacarusof/home/shared/Gaia/Coliseum/Delays/paper_code/Datasets/neurons_datasets/raster_delay_tuning_dataset.mat','raster','-v7.3')

%% Extract and save spikes 

% matrix of size N X (Tr x Rep) X time

spikes = NaN(length(complete_dataset),650,200);
for d = 1:length(complete_dataset)
    
    s = complete_dataset(d).spikeTimes;
    e = complete_dataset(d).eventTimes;
    binSize=0.01; %this determines how big the bins are - you can vary this from 0.0001 to whatever fits best
    window=[-1 1];
    
    which_trials = getWhichTrials(e);
    
    tot_spikes=[];
    for trx=1:length(which_trials)
        tr = which_trials(trx);
        events = e(:,1,tr);
        [~, ~, rasterX, rasterY, ~, ba] = psthAndBA(s, events, window, binSize);        
        % save the spikes 
        tot_spikes = [tot_spikes;ba];
    end
    % save the spikes
    spikes(d,:,:) = tot_spikes;
    d
end

%% Here we extract the infor we need for the running the python analysis code

% Let's append one thing at the time 
% add the spikes to the dataset
delay_tuning_dataset.spikes  = spikes;

%2. save info about the window for the spikes and the binsize
delay_tuning_dataset.window_spikes = window;
delay_tuning_dataset.binSize = binSize;

%3. save info about the trial order
delay_tuning_dataset.trials = [1,2,3,4,5,6,7,8,9,10,11,12,13];

%4. save if responsive (0 or 1)
delay_tuning_dataset.resp = extractfield(complete_dataset,'responsive')';

%5. no need to load the IFR, I can just load the whole variable on python 

%6. 3D coord
delay_tuning_dataset.coord3D = reshape(extractfield(complete_dataset,'coord3D'),3,[])';

%12. peak FR
delay_tuning_dataset.peaks = reshape(extractfield(complete_dataset,'peakFR'),13,[])';

%13 coords lims - these were extracted from Brainglobe
delay_tuning_dataset.ML_lim = [3639.00311745, 5683.28581603];
delay_tuning_dataset.AP_lim = [ 8311.3487132 , 10110.67790772];
delay_tuning_dataset.depth_lim = [ 994.70347906, 3387.01107207];

%14 save the modality 
delay_tuning_dataset.modality = extractfield(complete_dataset,'modality')';

% bootstrap output
delay_tuning_dataset.all_boot_aud = reshape(extractfield(complete_dataset,'sig_aud'),11,[])';
delay_tuning_dataset.all_boot_vis = reshape(extractfield(complete_dataset,'sig_vis'),11,[])';

% animal ID and experiment ID
delay_tuning_dataset.animal_ID= extractfield(complete_dataset,'Animal_ID')';
delay_tuning_dataset.experiment_ID = extractfield(complete_dataset,'Experiment_ID')';

% p values for each condition 
temp = reshape(extractfield(complete_dataset,'Z_pval'),14,[])';
delay_tuning_dataset.pvals = temp;

% proceed to save it
file_path = fullfile('/camp/lab/iacarusof/home/shared/Gaia/Coliseum/Delays/paper_code/Datasets/neurons_datasets/', ['delay_tuning_dataset' '.mat']);
save(file_path,'delay_tuning_dataset','-v7.3')
