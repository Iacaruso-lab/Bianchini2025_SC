%% Analysis of CCG for Coliseum data

% load the whole dataset and extract the info abou thte experiment ID (this is our session number)
neuron_matrix1 = load('neuron_matrix'); 
neuron_matrix1 = neuron_matrix1.neuron_matrix;
experiment_ID1 = extractfield(neuron_matrix1,'Experiment_ID');
sessions1 = unique(experiment_ID1);

neuron_matrix2 = load('neuron_matrix_more_neurons'); 
neuron_matrix2 = neuron_matrix2.neuron_matrix;
experiment_ID2 = extractfield(neuron_matrix2,'Experiment_ID');
sessions2 = unique(experiment_ID2);

sessions = [sessions1,sessions2]';

%% run the CCG analysis

camp = 1;
if camp == 1
    output_dir = "/nemo/lab/iacarusof/home/shared/Gaia/Coliseum/Delays/save_load_datasets/CCG_output/";
else
    output_dir = "Z:\home\shared\Gaia\Coliseum\Delays\save_load_datasets\CCG_output\";
end

name = "ccg_output_100us_ses";
i = ID;
this_session = sessions(i);
disp("processing session: " + this_session);
% Find the indices where experiment_ID is equal to i
indices = find([neuron_matrix.Experiment_ID] == this_session);

% initiate empty array
n_rep = 50;
binSize=0.0001; %this determines how big the bins are - you can vary this from 0.0001 to whatever fits best
window=[0.5 2.5]; % I have 3s of inter trial time

n_neurons = length(indices);
n_trials = size(neuron_matrix(indices(1)).eventTimes,3)*n_rep;
n_times = (window(2)-window(1)) /binSize;

spikes = nan(n_neurons,n_trials,n_times);

for n = 1:length(indices)% iterate through neurons of the experiment
    this_n = indices(n);

    s = neuron_matrix(this_n).spikeTimes; % get the spike times
    e = neuron_matrix(this_n).eventTimes; % get the event times
    [~, ~, ~, ~, ~, ba] = psthAndBA(s, e, window, binSize); % ba in binned spike times

    spikes(n,:,:) = ba;
end

spikes = permute(spikes,[2,3,1]); % change it to Trials x time x N neurons

% start calculating CCG

% convert spike train to 'has at least 1 spike in bin'
spikes = logical(spikes); 

% at the moment take the whole stimulus window
% get times of interest
start_time = 1;
end_time = size(spikes,2);
spikes_subset = spikes(:, start_time:end_time, :); % if you every want to use a subset of spikes

ccg_output = struct;

% jitter spiketrains
jit_window = 100; %10ms 
for j = 1:n_neurons %to the number of neurons
    data_jitter{j} = jitter(spikes_subset(:,:,j),jit_window);
    data_real{j} = spikes_subset(:,:,j);
end

% get neuron id pairs
for j = 1:n_neurons %for each neuron
    ccgs{j}.neuron_id_pairs = nan(2*(n_neurons-j),2);
    cnt = 0;
    for k = j+1:n_neurons
        % pre & post comps
        for jj = 1:2
            cnt = cnt + 1;
            if jj == 1 % first neuron is pre
                ccgs{j}.neuron_id_pairs(cnt,:) = [j,k];
            else % first neuron is post
                ccgs{j}.neuron_id_pairs(cnt,:) = [k,j];
            end
        end
    end
end

% aggregate ccgs into ccg output struct
fields = fieldnames(ccgs{1});
for j = 1:length(fields)
    ccg_output.(fields{j}) = ccgs{1}.(fields{j});
end
neuron_id_pairs = reshape(extractfield(ccg_output,'neuron_id_pairs'),[],2);

% select only responsive neurons
% extract modality of all neurons in this session 
modality = extractfield(neuron_matrix,'modality');
modality = modality(indices);
to_do = find(modality>0); % if you only want to perform the analysis on the responsive neurons

% here start calculating the CCG
max_lag = 150; %15ms
min_lag = -150; %15ms
parfor j = 1:n_neurons %for each neuron
    
        ccgs{j}.ccg_norm = nan(2*(n_neurons-j),max_lag-min_lag+1);
        ccgs{j}.ccg_unnorm = nan(2*(n_neurons-j),max_lag-min_lag+1); 
        ccgs{j}.ccg_norm_jitter = nan(2*(n_neurons-j),max_lag-min_lag+1);
        ccgs{j}.ccg_unnorm_jitter = nan(2*(n_neurons-j),max_lag-min_lag+1);
        ccgs{j}.neuron_id_pairs = nan(2*(n_neurons-j),2);

        disp("processing neuron: " + j);
        cnt = 0;
        for k = j+1:n_neurons
            % pre & post comps
            for jj = 1:2
                cnt = cnt + 1;
                if jj == 1 % first neuron is pre
                    ccgs{j}.neuron_id_pairs(cnt,:) = [j,k];
                else % first neuron is post
                    ccgs{j}.neuron_id_pairs(cnt,:) = [k,j];
                end

                %if ismember(k,to_do) && ismember(j,to_do) % to reduce time only do the responsive ones
                    if jj == 1 % first neuron is pre
                        [ccgs{j}.ccg_norm(cnt,:), ccgs{j}.ccg_unnorm(cnt,:)] = xcorr_gm(data_real{j}, data_real{k}, max_lag, min_lag);
                        [ccgs{j}.ccg_norm_jitter(cnt,:), ccgs{j}.ccg_unnorm_jitter(cnt,:)]  = xcorr_gm(data_jitter{j}, data_jitter{k}, max_lag, min_lag);
                    else % first neuron is post
                        [ccgs{j}.ccg_norm(cnt,:), ccgs{j}.ccg_unnorm(cnt,:)] = xcorr_gm(data_real{k},data_real{j}, max_lag, min_lag);
                        [ccgs{j}.ccg_norm_jitter(cnt,:), ccgs{j}.ccg_unnorm_jitter(cnt,:)] = xcorr_gm(data_jitter{k},data_jitter{j}, max_lag, min_lag);
                    end
                %end
                
            end
        end
end


% aggregate ccgs into ccg output struct
fields = fieldnames(ccgs{1});
for j = 1:length(fields)
    ccg_output.(fields{j}) = ccgs{1}.(fields{j});
end

for i = 2:n_neurons-1
    for j = 1:length(fields)
        ccg_output.(fields{j}) = [ccg_output.(fields{j}); ccgs{i}.(fields{j})];
    end
end

% ccg control is base - jitter
ccg_output.ccg_control = ccg_output.ccg_norm-ccg_output.ccg_norm_jitter;
ccg_output.ccg_control_unnorm = ccg_output.ccg_unnorm-ccg_output.ccg_unnorm_jitter;

% store the spike data as well
ccg_output.data = spikes;

% and save all
save(output_dir + name + int2str(this_session)+".mat", 'ccg_output');

%% load the output of the CCG analysis

camp = 0;
if camp == 1
    output_dir = "/nemo/lab/iacarusof/home/shared/Gaia/Coliseum/Delays/save_load_datasets/CCG_output_base/shuffled/depth/";
else
    output_dir = "Z:\home\shared\Gaia\Coliseum\Delays\save_load_datasets\CCG_output_base\shuffled\depth\";
end

% cropped = 1;
res = 0.1;%1;
name = "ccg_output_100us_rep";
% name = "ccg_output_1ms_ses";

name_save = 'all_ccg_100us';

%% description of the struct fields

% all_ccg = struct();
% all_ccg.sess_n = []; % session number
% all_ccg.pair_ids = []; % pair ids
% all_ccg.peaks = []; % peaks of CCG
% all_ccg.troughs = []; % troughs of CCG
% all_ccg.peak_lag = []; % peak lags of CCG
% all_ccg.trough_lag = []; % troughs lags of CCG
% all_ccg.sig_idx = []; % sig idx after criteria
% all_ccg.pre_id = []; %id on pre neurons
% all_ccg.post_id =[]; %id of post neuron
% all_ccg.pair_positions_depth=[];% position of pairs in depth
% all_ccg.pair_distance_depth=[]; %distance between pairs in depth
% all_ccg.pre_pair_positions_3D = []; %position of the pre in 3D - ['AP','depth_in_brain','ML']
% all_ccg.post_pair_positions_3D = []; %position of the post in 3D - ['AP','depth_in_brain','ML']
% all_ccg.pair_3D_dist=[]; % 3D distance of pairs
% all_ccg.pre_modality = []; % modality of pre neuron
% all_ccg.post_modality = []; % modality of post neuron
% all_ccg.pair_modality = []; % modality of both neurons
% all_ccg.pair_meanML_pos= []; % mean position of each pair along the ML axis
% all_ccg.pair_meanAP_pos= []; % mean position of each pair along the AP axis
% all_ccg.pair_meandepth_pos= []; %% mean position of each pair along the depth axis
% all_ccg.pre_mean_FR = []; %FR in Hz between 50ms and 500ms after stimulus onset of the PRE neuron
% all_ccg.post_mean_FR = []; %FR in Hz between 50ms and 500ms after stimulus onset of the POST neuron
% all_ccg.corrected_CCG = []; %CCG

%%
% Define field names
fieldNames = {'sess_n', 'pair_ids', 'peaks', 'troughs', 'peak_lag', ...
              'trough_lag', 'sig_idx_4sd','sig_idx_5sd','sig_idx_6sd','sig_idx_7sd', 'pre_id', 'post_id', ...
              'pair_positions_depth', 'pair_distance_depth', ...
              'pre_pair_positions_3D', 'post_pair_positions_3D', ...
              'pair_3D_dist', 'pre_modality', 'post_modality', ...
              'pair_modality', 'pair_meanML_pos', 'pair_meanAP_pos', ...
              'pair_meandepth_pos', 'pre_mean_FR', 'post_mean_FR', ...
              'corrected_CCG'};

% Create the empty structure
all_ccg = cell2struct(cell(size(fieldNames)), fieldNames, 2);


% initial number of pairs
n_pairs = 0;
n_reps = 101;
for i = 1:length(sessions) % iterate through sessions
   
    this_session = sessions(i);

    if ismember(this_session,sessions1)
        neuron_matrix = neuron_matrix1;
    elseif ismember(this_session,sessions2)
        neuron_matrix = neuron_matrix2;
    else
        disp('There is a problem here!')
    end

    disp("processing session: " + this_session);

    % load the file
    load(output_dir + name + int2str(this_session)+".mat");

    % this is the nromalised and jitter corrected CCG
    ccg_control = ccg_output.ccg_control;
    
    % also append the CCG
    all_ccg.corrected_CCG = [all_ccg.corrected_CCG ; ccg_control];

    % get number of pairs in session
    n_pairs = n_pairs + size(ccg_control,1);

    % fill out the struct all together
    all_ccg.sess_n = [all_ccg.sess_n;this_session*ones(size(ccg_control,1),1)];

    % pair ids
    pair_ids = ccg_output.neuron_id_pairs;

    if i == 1
        max_n = 0;
    else
        max_n = max(all_ccg.pair_ids(:,1));
    end
    all_ccg.pair_ids = [all_ccg.pair_ids ; ccg_output.neuron_id_pairs + max_n];

    % get peaks and troughs
    [peaks, lpeak_lag] = max(ccg_control,[],2);
    [troughs, ltrough_lag] = min(ccg_control,[],2);
    % get peak lags
    if res == 0.1
        lag_val = 150; % us
    else
        lag_val = 15; % ms
    end
    
    peak_lag = lag_val-lpeak_lag+1;
    peak_lag = peak_lag*-1; % important!
    trough_lag = lag_val-ltrough_lag+1;
    trough_lag = trough_lag*-1;
    
    % get significant ccgs
    if res == 0.1
        extremes = 50; % ms
    else
        extremes = 5; % ms
    end
    
    noise_distribution = [ccg_control(:, 1:extremes), ccg_control(:, end-extremes:end)]; %flanks of the jittered-corrected CCG
    noise_distribution(2:2:end,:) = noise_distribution(1:2:end-1,:);
    noise_std = std(noise_distribution,0,2, 'omitnan');
    noise_mean = mean(noise_distribution,2);
    
    sig_min_std = 0;

    if res == 0.1
        sig_max_lag = 50; % ms
    else
        sig_max_lag = 5; % ms
    end
    these_sd = [4,5,6,7]; % possible SD values
    for s = 1:length(these_sd)

        sig_num_stds = these_sd(s);
        sig_idx = (noise_std>sig_min_std ) & ...
            (peaks>(sig_num_stds*noise_std + noise_mean)) & ...
            (abs(peak_lag) <= sig_max_lag);

        length(find(sig_idx==1)) %display it to get an idea
        
        field_name = ['sig_idx_' num2str(sig_num_stds) 'sd'];
        all_ccg.(field_name) = [all_ccg.(field_name); sig_idx];
    end

    % append everything to the struct
    all_ccg.peaks = [all_ccg.peaks ; peaks];
    all_ccg.troughs = [all_ccg.troughs;troughs];
    all_ccg.peak_lag = [all_ccg.peak_lag;peak_lag];
    all_ccg.trough_lag = [all_ccg.trough_lag;trough_lag];
    

    % get position of each pair of neuron

    % which one is pre and which is post
    pre_id = pair_ids(:,1);
    post_id = pair_ids(:,2);

    % get a FR for all the trace
    mean_FR = squeeze(mean(mean(ccg_output.data(:,:,:))))/res*1000;
    all_ccg.pre_mean_FR = [all_ccg.pre_mean_FR;mean_FR(pre_id)];
    all_ccg.post_mean_FR = [all_ccg.post_mean_FR;mean_FR(post_id)];

    % Find the indices where experiment_ID is equal to i
    indices = find([neuron_matrix.Experiment_ID] == this_session);
    nan_indices = isnan(pair_ids); % Find indices with NaN values
    valid_indices = find(~nan_indices); % Find indices without NaN values
    
    nan_indices = isnan(pre_id); % Find indices with NaN values
    pre_valid_indices = find(~nan_indices); % Find indices without NaN values
    nan_indices = isnan(post_id); % Find indices with NaN values
    post_valid_indices = find(~nan_indices); % Find indices without NaN values

    % extract position of all neurons in this session 
    depth_inSC = extractfield(neuron_matrix,'depth_inSC');
    depth_inSC = depth_inSC(indices);    
    pair_positions_depth = [depth_inSC(pre_id(pre_valid_indices));depth_inSC(post_id(post_valid_indices))]';
    pair_distance_depth  = abs(pair_positions_depth(:,1)-pair_positions_depth(:,2));

    % store depth in SC info
    all_ccg.pre_id = [all_ccg.pre_id ; pre_id + max_n];
    all_ccg.post_id = [all_ccg.post_id ; post_id + max_n];
    all_ccg.pair_positions_depth =[all_ccg.pair_positions_depth; pair_positions_depth];
    all_ccg.pair_distance_depth=[all_ccg.pair_distance_depth;pair_distance_depth];
    
    % estimate the distance in 3D 
    % extract position of all neurons in this session 
    coord3D = reshape(extractfield(neuron_matrix,'coord3D'),3,[]);
    coord3D = coord3D(:,indices);
    pre_pair_positions_3D = coord3D(:,pre_id(pre_valid_indices));   
    post_pair_positions_3D = coord3D(:,post_id(post_valid_indices));   
    pair_3D_dist=sqrt(sum((pre_pair_positions_3D - post_pair_positions_3D) .^ 2));

    % save actual ML,AP,depth coordinates in the brain
    all_ccg.pre_pair_positions_3D = [all_ccg.pre_pair_positions_3D ; pre_pair_positions_3D'];
    all_ccg.post_pair_positions_3D = [all_ccg.post_pair_positions_3D ; post_pair_positions_3D'];
    all_ccg.pair_3D_dist = [all_ccg.pair_3D_dist ; pair_3D_dist'];

    % also output the mean location in the ML,AP, and depth axes
    pair_meanML_pos = (pre_pair_positions_3D(3,:) + post_pair_positions_3D(3,:))/2;
    pair_meanAP_pos = (pre_pair_positions_3D(1,:) + post_pair_positions_3D(1,:))/2;
    pair_meandepth_pos = (pre_pair_positions_3D(2,:) + post_pair_positions_3D(2,:))/2;
    
    % and save it 
    all_ccg.pair_meanML_pos = [all_ccg.pair_meanML_pos ; pair_meanML_pos'];
    all_ccg.pair_meanAP_pos = [all_ccg.pair_meanAP_pos ; pair_meanAP_pos'];
    all_ccg.pair_meandepth_pos = [all_ccg.pair_meandepth_pos ; pair_meandepth_pos'];

    % extract modality of all neurons in this session 
    modality = extractfield(neuron_matrix,'modality');
    modality = modality(indices);
    pair_modality = [modality(pre_id(pre_valid_indices));modality(post_id(post_valid_indices))]';
    pre_modality = pair_modality(:,1);
    post_modality = pair_modality(:,2);

    % save the modality of each neuron
    all_ccg.pre_modality = [all_ccg.pre_modality ; pre_modality];
    all_ccg.post_modality = [all_ccg.post_modality ; post_modality];
    all_ccg.pair_modality = [all_ccg.pair_modality ; pair_modality];

end

% and save all
save(output_dir + name_save +".mat", 'all_ccg','-v7.3');

%% load the output of the CCG analysis - SHUFFLED version

camp = 0;
if camp == 1
    output_dir = "/nemo/lab/iacarusof/home/shared/Gaia/Coliseum/Delays/save_load_datasets/CCG_output_base/shuffled/depth/";
else
    output_dir = "Z:\home\shared\Gaia\Coliseum\Delays\save_load_datasets\CCG_output_base\shuffled\depth\";
end

% cropped = 1;
res = 0.1;%1;
name = "ccg_output_100us_rep";

name_save = 'all_ccg_100us';

%%
% Define field names
fieldNames = {'rep_n', 'pair_ids', 'peaks', 'troughs', 'peak_lag', ...
              'trough_lag', 'sig_idx_4sd','sig_idx_5sd','sig_idx_6sd','sig_idx_7sd', 'pre_id', 'post_id', ...
              'pair_positions_depth', 'pair_distance_depth', ...
              'pre_pair_positions_3D', 'post_pair_positions_3D', ...
              'pair_3D_dist', 'pre_modality', 'post_modality', ...
              'pair_modality', 'pair_meanML_pos', 'pair_meanAP_pos', ...
              'pair_meandepth_pos', 'pre_mean_FR', 'post_mean_FR', ...
              'corrected_CCG'};

% Create the empty structure
% all_ccg = cell2struct(cell(size(fieldNames)), fieldNames, 2);


for i = 76:length(sessions) % iterate through sessions

    this_rep = i;
    disp("processing rep: " + this_rep);
    file_path = fullfile(output_dir + name + int2str(this_rep)+".mat");

%     if i<=72
       neuron_matrix = neuron_matrix1;
%     else
%         neuron_matrix = neuron_matrix2;
%     end

    if isfile(file_path)
        % The file exists, so you can load it
        load(file_path);

        % this is the nromalised and jitter corrected CCG
        ccg_control = ccg_output.ccg_control;

        % also append the CCG
        all_ccg.corrected_CCG = [all_ccg.corrected_CCG ; ccg_control];

        % fill out the struct all together
        all_ccg.rep_n = [all_ccg.rep_n;this_rep*ones(size(ccg_control,1),1)];

        % pair ids - you need to transform them back to the real neuron id
        real_ids = ccg_output.which_n(:,2);
        pair_ids = ccg_output.neuron_id_pairs;
        for s = 1:size(pair_ids,2)
            for v = 1:length(real_ids)
                pair_ids(pair_ids(:,s) == v,s) = real_ids(v);
            end
        end
        all_ccg.pair_ids = [all_ccg.pair_ids ; pair_ids];

        % get peaks and troughs
        [peaks, lpeak_lag] = max(ccg_control,[],2);
        [troughs, ltrough_lag] = min(ccg_control,[],2);
        % get peak lags
        if res == 0.1
            lag_val = 150; % us
        else
            lag_val = 15; % ms
        end

        peak_lag = lag_val-lpeak_lag+1;
        peak_lag = peak_lag*-1;
        trough_lag = lag_val-ltrough_lag+1;
        trough_lag = trough_lag*-1;

        % get significant ccgs
        if res == 0.1
            extremes = 50; % ms
        else
            extremes = 5; % ms
        end

        noise_distribution = [ccg_control(:, 1:extremes), ccg_control(:, end-extremes:end)]; %flanks of the jittered-corrected CCG
        noise_distribution(2:2:end,:) = noise_distribution(1:2:end-1,:);
        noise_std = std(noise_distribution,0,2, 'omitnan');
        noise_mean = mean(noise_distribution,2);

        sig_min_std = 0;

        if res == 0.1
            sig_max_lag = 50; % ms
        else
            sig_max_lag = 5; % ms
        end
        these_sd = [4,5,6,7]; % possible SD values
        for s = 1:length(these_sd)

            sig_num_stds = these_sd(s);
            sig_idx = (noise_std>sig_min_std ) & ...
                (peaks>(sig_num_stds*noise_std + noise_mean)) & ...
                (abs(peak_lag) <= sig_max_lag);

            length(find(sig_idx==1)) %display it to get an idea

            field_name = ['sig_idx_' num2str(sig_num_stds) 'sd'];
            all_ccg.(field_name) = [all_ccg.(field_name); sig_idx];
        end

        % append everything to the struct
        all_ccg.peaks = [all_ccg.peaks ; peaks];
        all_ccg.troughs = [all_ccg.troughs;troughs];
        all_ccg.peak_lag = [all_ccg.peak_lag;peak_lag];
        all_ccg.trough_lag = [all_ccg.trough_lag;trough_lag];


        % get position of each pair of neuron

        % which one is pre and which is post
        pre_id = pair_ids(:,1);
        post_id = pair_ids(:,2);
        pre_id_pos = ccg_output.neuron_id_pairs(:,1);
        post_id_pos = ccg_output.neuron_id_pairs(:,2);

        % get a FR for all the trace
        mean_FR = squeeze(mean(mean(ccg_output.data(:,:,:))))/res*1000;
        all_ccg.pre_mean_FR = [all_ccg.pre_mean_FR;mean_FR(pre_id_pos)];
        all_ccg.post_mean_FR = [all_ccg.post_mean_FR;mean_FR(post_id_pos)];

        % extract position of all neurons in this session
        depth_inSC = extractfield(neuron_matrix,'depth_inSC');
        non_zero_pre_id = pre_id ~= 0;
        new_array_pre = zeros(length(pre_id),1);
        new_array_pre(non_zero_pre_id) = depth_inSC(pre_id(non_zero_pre_id));

        non_zero_post_id = post_id ~= 0;
        new_array_post = zeros(length(post_id),1);
        new_array_post(non_zero_post_id) = depth_inSC(post_id(non_zero_post_id));

        pair_positions_depth = [new_array_pre,new_array_post];
        pair_distance_depth  = abs(pair_positions_depth(:,1)-pair_positions_depth(:,2));

        % store depth in SC info
        all_ccg.pre_id = [all_ccg.pre_id ; pre_id];
        all_ccg.post_id = [all_ccg.post_id ; post_id];
        all_ccg.pair_positions_depth =[all_ccg.pair_positions_depth; pair_positions_depth];
        all_ccg.pair_distance_depth=[all_ccg.pair_distance_depth;pair_distance_depth];

        % estimate the distance in 3D
        % extract position of all neurons in this session
        coord3D = reshape(extractfield(neuron_matrix,'coord3D'),3,[]);
        pre_pair_positions_3D = zeros(3,length(pre_id));
        pre_pair_positions_3D(:,non_zero_pre_id) = coord3D(:,pre_id(non_zero_pre_id));

        post_pair_positions_3D = zeros(3,length(post_id));
        post_pair_positions_3D(:,non_zero_post_id) = coord3D(:,post_id(non_zero_post_id));

        pair_3D_dist=sqrt(sum((pre_pair_positions_3D - post_pair_positions_3D) .^ 2));

        % save actual ML,AP,depth coordinates in the brain
        all_ccg.pre_pair_positions_3D = [all_ccg.pre_pair_positions_3D ; pre_pair_positions_3D'];
        all_ccg.post_pair_positions_3D = [all_ccg.post_pair_positions_3D ; post_pair_positions_3D'];
        all_ccg.pair_3D_dist = [all_ccg.pair_3D_dist ; pair_3D_dist'];

        % also output the mean location in the ML,AP, and depth axes
        pair_meanML_pos = (pre_pair_positions_3D(3,:) + post_pair_positions_3D(3,:))/2;
        pair_meanAP_pos = (pre_pair_positions_3D(1,:) + post_pair_positions_3D(1,:))/2;
        pair_meandepth_pos = (pre_pair_positions_3D(2,:) + post_pair_positions_3D(2,:))/2;

        % and save it
        all_ccg.pair_meanML_pos = [all_ccg.pair_meanML_pos ; pair_meanML_pos'];
        all_ccg.pair_meanAP_pos = [all_ccg.pair_meanAP_pos ; pair_meanAP_pos'];
        all_ccg.pair_meandepth_pos = [all_ccg.pair_meandepth_pos ; pair_meandepth_pos'];

        % extract modality of all neurons in this session
        modality = extractfield(neuron_matrix,'modality');
        pre_modality = zeros(length(pre_id),1);
        pre_modality(non_zero_pre_id) = modality(pre_id(non_zero_pre_id));

        post_modality = zeros(length(post_id),1);
        post_modality(non_zero_post_id) = modality(post_id(non_zero_post_id));

        pair_modality = [pre_modality,post_modality];
        % save the modality of each neuron
        all_ccg.pre_modality = [all_ccg.pre_modality ; pre_modality];
        all_ccg.post_modality = [all_ccg.post_modality ; post_modality];
        all_ccg.pair_modality = [all_ccg.pair_modality ; pair_modality];
    else
        % The file doesn't exist, so skip this
    end
end

% and save all
save(output_dir + name_save +".mat", 'all_ccg','-v7.3');