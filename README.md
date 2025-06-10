# Functional specialisation of multisensory temporal integration in the mouse superior colliculus
Code to reproduce the figures from Bianchini et al., "Functional specialisation of multisensory temporal integration in the mouse superior colliculus" available at https://www.biorxiv.org/content/10.1101/2025.02.11.637674v3

Data is available on Figshare doi: 10.25418/crick.28685360.

## :woman_technologist: Getting started
1. Clone the repository
2. Download the data (doi: 10.25418/crick.28685360)
3. Create a [conda environment](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html) with necessary packages
```
conda create -n my_env python=3.8.16 numpy=1.23.5 pandas=1.5.2 scipy=1.10.0 matplotlib seaborn jupyter
```
4. Start jupyter in this conda environment and run the desired script
```
conda activate my_env
jupyter notebook
```
## Data structure

The data is organised into four subfolders: 
1. neurons_datasets
2. decoder_datasets
3. connectivity_datasets
4. movement_control_datasets

Each folder contains the main datasets along with intermediate datasets obtained at different stages of pre-processing. Each dataset is clearly referenced in the corresponding analysis step.

## Repo structure

The repository consists of three main folders:
1. Analysis
2. Figures_code
3. Figures_output

The **Analysis folder** is further divided into five subfolders:
1. neurons_analysis
2. decoder_analysis
3. connectivity_analysis
4. movement_control_analysis
5. helper_functions

Each folder contains a main Jupyter notebook (```Main_analysis_code```) and additional scripts to run parts of the analysis. These scripts are referenced in the main Jupyter notebook.
The **Figures_code** folder contains the code needed to reproduce each figure from the paper. The figures are then saved in the **Figures_output** folder.

## Dependencies

- ZETA test: https://github.com/JorritMontijn/zetatest
- https://github.com/cortex-lab/spikes
