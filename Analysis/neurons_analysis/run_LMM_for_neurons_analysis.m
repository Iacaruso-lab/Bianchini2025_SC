%% Code to run all LMM mentioned in the figure codes

%% 1. Differences in depth distribution based on modality of the neurons

% load the dataframe

clc
clear all
df = readtable('depth_distribution_neurons.csv');

% run model

% Convert modality to categorical
df.modality = categorical(df.modality);

% Set 'non-responsive' as the reference level
df.modality = reordercats(df.modality, ...
    {'non-responsive', 'visual', 'auditory', 'audio-visual', 'gated'});

% Optional: convert IDs to categorical if needed
df.animal_ID = categorical(df.animal_ID);
df.experiment_ID = categorical(df.experiment_ID);

formula = ['depth_norm ~ 1 + modality + (1|animal_ID) + (1|experiment_ID)'];
model1 = fitlme(df, formula);
res0 = model1.Coefficients;
results = anova(model1)

%% 2. Check for differences in anatomical distribution of neurons based on modality

% load the dataframe

clc
clear all
df = readtable('df_all_modalities.csv');

% run model
df.Modality = categorical(df.Modality);
df.Modality = reordercats(df.Modality, {'nr','auditory', 'visual', 'multisensory'});
formula = ['AP ~ Modality + (1 | Animal)']
model1 = fitlme(df, formula) 

formula = ['ML ~ Modality + (1 | Animal)']
model1 = fitlme(df, formula) 

formula = ['Depth ~ Modality + (1 | Animal)']
model1 = fitlme(df, formula) 


%% 3. Check for differences in selectivity, FR modualtion index and reliability amongst delays

% load the dataframe

clc
clear all
df = readtable('delay_neurons_properties.csv');

%%
% run model
formula = ['reliability ~ 1 + pref_delay + (1|animal_ID) + (1|experiment_ID)'];
model1 = fitlme(df, formula)

formula = ['modulation_index ~ 1 + pref_delay + (1|animal_ID) + (1|experiment_ID)'];
model1 = fitlme(df, formula)

formula = ['selectivity ~ 1 + pref_delay ']%+ (1|animal_ID) + (1|experiment_ID)'];
model1 = fitlme(df, formula)

res0 = model1.Coefficients;
results = anova(model1)
idx = strcmp(res0.Name, 'pref_delay');
selectivity_slope = res0.Estimate(idx)

%% 4. Check relationship between observed and predicted (based on sum) preferred delay

% load the dataframe

clc
clear all
df = readtable('observed_vs_predicted_delay.csv');

% run model
formula = ['obs_delay ~ 1 + pred_delay + (1|animal_ID) + (1|experiment_ID)'];
model1 = fitlme(df, formula)

%% 5. Check whether MII varies betwen delays

% load the dataframe

clc
clear all
df = readtable('MII_data_for_lmm.csv');

% run model
formula = 'MII_value ~ delay + (1|neuron_ID) + (1|animal_ID) + (1|experiment_ID)';
model1 = fitlme(df, formula)

%% 6. Check whether MII varies betwen delays - including all delays before and after

% load the dataframe

clc
clear all
df = readtable('MII_data_for_all_delaysBA.csv');

% run model
formula = 'MII_value ~ delay + (1|neuron_ID) + (1|animal_ID) + (1|experiment_ID)';
model1 = fitlme(df, formula)
res0 = model1.Coefficients;
results = anova(model1)

%% 7. Check whether MII varies betwen delays: splitting delays between category before and after

% load the dataframe

clc
clear all
df = readtable('MII_data_onlyBA_delaysBA.csv');

% run model
formula = 'MII ~ condition + (1|animal_ID) + (1|experiment_ID)';
model1 = fitlme(df, formula)

