# Hippocampus_AP_Axis

[![DOI](https://zenodo.org/badge/134175281.svg)](https://zenodo.org/badge/latestdoi/134175281)


### Description
All code used for the Hippocampus Anterior/Posterior gene expression and neuroimaging analyses from the paper: https://www.biorxiv.org/content/10.1101/587071v1. 

Preprint doi: https://doi.org/10.1101/587071


Included are several Jupyter notebooks running through each of the analyses in the paper, showing the exact code and data used to prepare and run each analysis and sub-analysis, and generate the figures from the manuscript. The repository also includes a library containing functions built specifically for these analyses, and links and instructions for accessing data.

The Notebooks are divided thematically, such that each corresponds to a certain set of analyses. Due to the size of some of the files, you will need to run some of these notebooks (particularly NB1) in order to generate the data used in later notebooks. I tried to demarcate areas where a previous notebook must be run in order to complete an analysis. However, many aspects of each notebook can be run independently without running other Notebooks (except perhaps NB1).

The notebooks are as follows:

N1_PrepareABAData.ipynb --> Retrieve Allen Brain Atlas data and prepare DataFrames for all subsequent analyses.

N2_FittingModel.ipynb --> Analyses presented in and relating to Figures 1, S4 and Tables 1, S1

N3_Explore_Features.ipynb --> Analyses presented in and relating to Figures 2, S2, S3, S10 and Tables S2, S3 and S4

N4_CellType_Analyses.ipynb --> Analyses presented in and relating to Figure S5 and Table S5

N5_BrainSpan_Replication.ipynb --> Analyses presented in and relating to Figure 3

N6_HAGGIS.ipynb --> Analyses presented in and relating to Figures 4, S11

N7_Correlations_with_Imaging.ipynb --> Analyses presented in and relating to Figures 5, S6, S7, S9 and Table S6

N8_Cognition_metaanalysis.ipynb --> Analyses presented in and relating to Figures 6, S8

### Requirements

Running these notebooks will require Python 3.6 or above, Jupyter, and a number of Python packages. An environment containing all of the necessary elements is included in the git repo. However, you will need to download Anaconda to activate the environment. See below for installation instructions.

https://www.anaconda.com/distribution/

### Reproducing analyses on your machine

To reproduce these analyses on your computer, all you have to do is:
* Download Anaconda if you haven't aready.
* Use the environment.yml file to recreate my conda environment. For instructions, see here: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file
* Clone or download this entire repository. 
* To run a notebook, navigate to repo and run the notebooks: jupyter notebook [Notebook.ipynb]

It is possible that the reconstruction of the environment will fail, particularly if using Windows. If so, you can still reconstruct the environment manually, as the environment.yml file contains all the information regarding package version and download source.

### Troubleshooting
If there are any problems, please don't hesitate to raise an issue.

