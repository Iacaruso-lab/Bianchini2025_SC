%% This code ususally run on a cluster

%% get number of arrays
 
arg = GetCommandLineArgs;
ID = str2double(arg{4,1});

%% select current folder 

ID = ID+1;

%% now run the file

n_perm = 1000;
addpath('/camp/lab/iacarusof/home/shared/Gaia/Coliseum/Delays/ZETA/')
addpath('/camp/lab/iacarusof/home/shared/Gaia/Coliseum/Delays/spikes-master/')
addpath('/camp/lab/iacarusof/home/shared/Gaia/Coliseum/Delays/paper_code/Datasets/neurons_datasets/')
addpath('/camp/lab/iacarusof/home/shared/Gaia/Coliseum/Delays/paper_code/Analysis/helper_functions/')
load('delay_tuning_dataset.mat'); 

these_neurons = find(extractfield(neuron_matrix,'responsive')'==1);

%% Make groups to run it in subgroups 

D = 1;
x = 10;
tot_n = length(these_neurons);
groups = 1:x:tot_n;
groups = [groups;groups+x];

%% Run bootstrap analysis

%all the parameters
rep = ID;
tr_n = 11;
db=0.25; % duration
window_res = 250;
smooth=0.5; % smoothing factor
dt_ms = 1;
line=linspace(0,db+0.01,1/(dt_ms*10^-3));

parfor dx = groups(1,rep):groups(2,rep)
    d = these_neurons(dx);
    s = delay_tuning_dataset(d).spikeTimes;
    e = delay_tuning_dataset(d).eventTimes-0.01;

    [peak_dist] = make_peak_dist(d,s,e,n_perm);
    save_peak_dist1(peak_dist,d,D)

end


%% After extracting the bootstrap distribution run this part of the code

%% Check which files you have in a directory 

% Specify the path
path = '/nemo/lab/iacarusof/home/shared/Gaia/Coliseum/Delays/boot_all/delay';
% Get the directory listing
dir_info = dir(path);

% Filter out directories
directories = dir_info([dir_info.isdir]);

% Extract directory names
directory_names = {directories.name};

% Exclude '.' and '..' directories
directory_names = directory_names(~ismember(directory_names, {'.', '..'}));

% Filter directory names containing the word 'result'
result_directories = directory_names(contains(directory_names, 'result'));
n_tr = 7;
n_mod = 2;
% look for each neuron
for n = 1:length(these_neurons)
    this_n = these_neurons(n);    

    full_output_boot = zeros(n_mod, n_tr, 10000);

    % Loop through each result directory
    for i = 1:length(result_directories)
        result_directory_name = result_directories{i};
        result_directory_path = fullfile(path, result_directory_name);
        % Set the current working directory to the result directory path
        cd(result_directory_path);
        
        currentfilename = sprintf('output-%d.mat', this_n);
        load(currentfilename);

        for m = 1:n_mod
            for t = 1:n_tr
                distribution = output_boot(m, t).max;
                full_output_boot(m, t, (i-1)*1000 + 1 : i*1000) = distribution;            
            end
        end
    end
    % save it! 

    % Define the base directory
    dirPath = '/camp/lab/iacarusof/home/shared/Gaia/Coliseum/Delays/boot_all/delay_BA/all';
    %dirPath = 'Z:\home\shared\Gaia\Coliseum\Delays\boot_all\all';
    
    % Format the file name using sprintf
    fileName = sprintf('output-%d', this_n);
    
    % Define the extension
    extension = '.mat';
    
    % Construct the full file path
    matname = fullfile(dirPath, [fileName extension]);
    
    save(matname,'full_output_boot')
    this_n
end

%% And compare it to the actual FR difference 

cd(dirPath);
db=0.25; % duration
window_res = 250;
smooth=0.5; % smoothing factor
dt_ms = 1;
line=linspace(0,db+0.01,1/(dt_ms*10^-3));
a = 0;
b = 0.001;
r = (b-a).*rand(10000,1) + a;

% look for each neuron
for n = 1:length(neurons_sorted)
    this_n = neurons_sorted(n);    
    currentfilename = sprintf('output-%d.mat', this_n);
    load(currentfilename);
    
    s = delay_tuning_dataset(this_n).spikeTimes;
    e = delay_tuning_dataset(this_n).eventTimes-0.01;

    % initiate the bootstrap output
    output_boot = zeros(n_mod,n_tr);

    for mod = 1:n_mod

        if mod == 1
            U_events = e(:,1,end-2);
        else
            U_events = e(:,1,end-1);
        end
        
        for t = 1:n_tr
            list_of_peak_heights = squeeze(full_output_boot(mod,t,:));
            % calculate actual difference
            tr_trials = e(:,1,t); %pool together trials
    
            [~,~,sIFR1]=getIFR(s, U_events,db,smooth);
            [~,~,sIFR2]=getIFR(s,tr_trials,db,smooth);
    
            % if there are duplicates in time adjust for it
            [~, w] = unique( sIFR1.vecTime, 'stable' );
            duplicate_indices = setdiff( 1:numel(sIFR1.vecTime), w );
            sIFR1.vecTime(duplicate_indices)=sIFR1.vecTime(duplicate_indices)+r(duplicate_indices);
            
            [~, w] = unique(sIFR2.vecTime, 'stable' );
            duplicate_indices = setdiff( 1:numel(sIFR2.vecTime), w );
            sIFR2.vecTime(duplicate_indices)=sIFR2.vecTime(duplicate_indices)+r(duplicate_indices);

            %perform interpolation
            tr1=interp1(sIFR1.vecTime,sIFR1.vecRate,line,'linear');
            tr2=interp1(sIFR2.vecTime,sIFR2.vecRate,line,'linear');

            % find the max value for each
            [~,pos]=max(tr1);
            if pos<6
                max_tr1=mean(tr1(1:pos+5));
            elseif pos>6
                max_tr1=mean(tr1(pos-5:pos+5));
            end
            
            [~,pos]=max(tr2);
            if pos<6
                max_tr2=mean(tr2(1:pos+5));
            elseif pos>6
                max_tr2=mean(tr2(pos-5:pos+5));
            end
            actual_diff = max_tr2-max_tr1;% put them together in a list

            sig=[];
            sig(1)=prctile(list_of_peak_heights,2.5);
            sig(2)=prctile(list_of_peak_heights,97.5);

            % save only the output of the bootstrap
            if actual_diff<sig(1)
                output_boot(mod,t) = -1;
            elseif actual_diff>sig(2)
                output_boot(mod,t) = 1;
            else
                output_boot(mod,t) = 0;
            end
        end
    end
    n
    delay_tuning_dataset(this_n).output_boot = output_boot;
end
   
% save it 

save(file_path,'delay_tuning_dataset','-v7.3')
