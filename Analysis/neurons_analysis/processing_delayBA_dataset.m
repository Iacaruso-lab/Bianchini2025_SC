%% load you Data.Info file and select the number of the experiment
% new data from 82
close all
clear all
clc

on_cluster = 1;
which_dataset = 'delayBA_preprocessing'; % let's select the dataset
if on_cluster == 1
    file_path = fullfile('/camp/lab/iacarusof/home/shared/Gaia/Coliseum/Delays/paper_code/Datasets/neurons_datasets/pre_processing_datasets', [which_dataset '.mat']);
    addpath(genpath('/camp/lab/iacarusof/home/shared/Gaia/Coliseum/Delays/spikes-master')) % for converting to Phy
    addpath(genpath('/camp/lab/iacarusof/home/shared/Gaia/Coliseum/Delays/ZETA')) % for zeta testing
    addpath(genpath('/camp/lab/iacarusof/home/shared/Gaia/Coliseum/Delays/paper_code/Analysis/helper_functions')) % helper functions
else
    file_path = fullfile('Z:\home\shared\Gaia\Coliseum\Delays\paper_code\Datasets\neurons_datasets\pre_processing_datasets', [which_dataset '.mat']);
end

load(file_path)

%% To make it easier for the future realign all eventTimes to the first stimulus! 
% ONCE THIS HAS BEEN RUN SHOULD NOT BE RUN AGAIN!

for d=1:length(neuron_matrix)

    e = neuron_matrix(d).eventTimes;
    temp = e(:,:,1); % -100ms
    e(:,:,1) = temp-0.100;

    temp = e(:,:,2); % -50ms
    e(:,:,2) = temp-0.050;

    temp = e(:,:,3); % -25ms
    e(:,:,3) = temp-0.025;

    neuron_matrix(d).eventTimes = e;
    neuron_matrix(d).stimuli = {-0.100,	-0.050,	-0.025,	0,	0.025,	0.050,	0.100, 'V','A', 'blank'};

    d
end

%% Let's set up the ZETA testing - to test for each condition if there is a response

for d=1:length(neuron_matrix)

    s = neuron_matrix(d).spikeTimes;
    e = neuron_matrix(d).eventTimes -0.25; 
    db = 0.5;
    intSmoothSd=0.5; %1
    
    n_trials = size(e,3);

    test = [];
    for i=1:n_trials
        events = e(:,1,i);
        [dblZetaP] = getZeta(s,events,db);
        test=[test;dblZetaP];
    end

    neuron_matrix(d).Z_pval=test;
    d

end

%% Calculate IFR

for d=1:length(neuron_matrix)
   
    s = neuron_matrix(d).spikeTimes;
    e = neuron_matrix(d).eventTimes -0.25;
    db=0.5;
    intSmoothSd=1;
    dt_ms = 1;
    line=linspace(0,db+0.01,1/(dt_ms*10^-3));
    
    smooth_FR=zeros(size(e,3),1000);
    
    for i=1:size(e,3)
        events = e(:,1,i);
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
save('/camp/lab/iacarusof/home/shared/Gaia/Coliseum/Delays/paper_code/Datasets/neurons_datasets/psth_delayBA_tuning_dataset','neuron_IFR','-v7.3')

%% Extract which neurons are responsive 

% First you need to correct for multiple comparison
for d = 1:length(neuron_matrix)
    p_vals = neuron_matrix(d).Z_pval(1:end-1);

    % check max IFR if > 10
    IFR = neuron_IFR(d).mean;
    maxf  =max(IFR,[],2);
    
    if ~isempty(neuron_matrix(d).output_boot)
        neuron_matrix(d).sig_vis = neuron_matrix(d).output_boot(1,:);
        neuron_matrix(d).sig_aud = neuron_matrix(d).output_boot(2,:);
    end

    % first correct for all multiple comparison
    p_val_corr = pval_adjust(p_vals,'BH');
    m = sum(find(p_val_corr(1:7)<0.05));
    neuron_matrix(d).modality = 0;

    if any(p_val_corr<0.05) && max(maxf)>10
        neuron_matrix(d).responsive = 1;
        % get their modality
        if p_val_corr(end-1)<0.05 && p_val_corr(end)>0.05 && maxf(8)>10
            neuron_matrix(d).modality = 1; %visual
        elseif p_val_corr(end)<0.05 && p_val_corr(end-1)>0.05 && maxf(9)>10
            neuron_matrix(d).modality = 2; %auditory
        elseif p_val_corr(end)<0.05 && p_val_corr(end-1)<0.05 && maxf(9)>10 && maxf(8)>10
            neuron_matrix(d).modality = 3; %aud-vis
        elseif p_val_corr(end)>0.05 && p_val_corr(end-1)>0.05 && m>0 && any(maxf(1:7)>10)
            neuron_matrix(d).modality = 4; %multi
        end
    else
        neuron_matrix(d).responsive = 0;
        neuron_matrix(d).modality = 0; %non responsive
    end

end

save(file_path,'neuron_matrix','-v7.3')

%% Extract peakFR

%all the parameters
tr_n = 11;
db=0.25; % duration
smooth=0.5; % smoothing factor
dt_ms = 1;
line=linspace(0,db+0.01,1/(dt_ms*10^-3));

for d = 1:length(neuron_matrix)

    s = neuron_matrix(d).spikeTimes;
    e = neuron_matrix(d).eventTimes-0.01;

    peak_FR = zeros(size(e,3),1);

    for t = 1:size(e,3)

        tr_trials = e(:,1,t); %pool together trials        
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
    neuron_matrix(d).peakFR = peak_FR;  
d
end


%% Extract and save spikes 

% matrix of size N X (Tr x Rep) X time

spikes1ms = NaN(length(neuron_matrix),450,2000);
for d = 1:length(neuron_matrix)
    
    s = neuron_matrix(d).spikeTimes;
    e = neuron_matrix(d).eventTimes;
    binSize=0.001; %this determines how big the bins are - you can vary this from 0.0001 to whatever fits best
    window=[-1 1];

    tot_spikes=[];
    for tr=1:size(e,3)-1
        events = e(:,1,tr);
        [~, ~, rasterX, rasterY, ~, ba] = psthAndBA(s, events, window, binSize);        
        % save the spikes 
        tot_spikes = [tot_spikes;ba];
    end
    % save the spikes
    spikes1ms(d,:,:) = tot_spikes;
    d
end

%% save the raster

for d = 1:length(neuron_matrix)
    
    s = neuron_matrix(d).spikeTimes;
    e = neuron_matrix(d).eventTimes;
    binSize=0.001; %this determines how big the bins are - you can vary this from 0.0001 to whatever fits best
    window=[-1 1];

    tot_spikes=[];
    for tr=1:size(e,3)-1
        events = e(:,1,tr);
        [~, ~, rasterX, rasterY, ~, ba] = psthAndBA(s, events, window, binSize);        
        % save the raster
        raster(d).tr(tr).x=rasterX;
        raster(d).tr(tr).y=rasterY;
    end

    d
end

save('/camp/lab/iacarusof/home/shared/Gaia/Coliseum/Delays/paper_code/Datasets/neurons_datasets/raster_delayBA_tuning_dataset.mat','raster','-v7.3')

%% Save it as a dataset to upload in python.

delayBA_dataset.stimuli = neuron_matrix(1).stimuli;
% 3D coord
delayBA_dataset.coord3D = reshape(extractfield(neuron_matrix,'coord3D'),3,[])';
%save if responsive (0 or 1)
delayBA_dataset.resp = extractfield(neuron_matrix,'responsive')';

%peak FR
delayBA_dataset.peaks = reshape(extractfield(neuron_matrix,'peakFR'),10,[])';
%save the modality 
delayBA_dataset.modality = extractfield(neuron_matrix,'modality')';
%save the responsive
delayBA_dataset.responsive = extractfield(neuron_matrix,'responsive')';
% bootstrap output
delayBA_dataset.all_boot_aud = reshape(extractfield(neuron_matrix,'sig_aud'),7,[])';
delayBA_dataset.all_boot_vis = reshape(extractfield(neuron_matrix,'sig_vis'),7,[])';

% animal ID and experiment ID
delayBA_dataset.animal_ID= extractfield(neuron_matrix,'Animal_ID')';
delayBA_dataset.experiment_ID = extractfield(neuron_matrix,'Experiment_ID')';
% spikes 
delayBA_dataset.spikes  = spikes;

% spikes 
delayBA_dataset.spikes5ms  = spikes5ms;
file_path = fullfile('camp/lab/iacarusof/home/shared/Gaia/Coliseum/Delays/paper_code/Datasets/neurons_datasets', ['delayBA_dataset' '.mat']);
save(file_path,'delayBA_dataset','-v7.3')

