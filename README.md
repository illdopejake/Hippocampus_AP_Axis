# Hippocampus_AP_Axis

### Description
All code used for the Hippocampus Anterior/Posterior gene expression analyses from the paper: [LINK TO PAPER COMING SOON]. 

Included are several Jupyter notebooks running through each of the analyses in the paper, showing the exact code and data used to prepare and run each analysis and sub-analysis, and generate the figures from the manuscript. The repository also includes a library containing functions built specifically for these analyses, and links and instructions for accessing data.

The Notebooks are divided thematically, such that each corresponds to a certain set of analyses. Due to the size of some of the files, you will need to run some of these notebooks (particularly NB1) in order to generate the data used in later notebooks. I tried to demarcate areas where a previous notebook must be run in order to complete an analysis. However, many aspects of each notebook can be run independently without running other Notebooks (except perhapse NB1).

The notebooks are as follows:

N1_PrepareABAData.ipynb --> Retrieve Allen Brain Atlas data and prepare DataFrames for all subsequent analyses.

N2_FittingModel.ipynb --> Analyses presented in and relating to Figure 1 and Table 1

N3_Explore_Features.ipynb --> Analyses presented in and relating to Figure 2 and Tables S2 and S3

N4_Correlations_with_Imaging.ipynb --> Analyses presented in and relating to Figures 3,4

### Requirements

### Reproducing analyses on your machine
