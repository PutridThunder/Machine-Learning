import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

###################################################################################
# python was being weird, had the dataset and file in the same folder, but it could not find the dataset, said it didnt exist

# moved the data into a subfolder that exists inside the project folder, used this to find the dataset

PROJECT_ROOT = Path(__file__).parent 
DATA_PATH = PROJECT_ROOT / "Data" / "AirQualityUCI.xlsx"

dataset = pd.read_excel(DATA_PATH) # convert the dataset into a pandas dataframe

######################################################################################################################################################################
# helper functions

def Section(text):
    print(f"\n================================================================================{text}================================================================================\n")

def Dset_info(x):
    """function for observing the info in a given dataframe"""
    print(f"shape: {x.shape}") # columns, row
    x.info() # column names, non null index counts, Dtype, memory usage
    print(x.describe()) # for each column, filled index count, mean, min, 25%, 50%, 75%, max, std

def Plot_histogram(data):
    data.hist(bins= 50, figsize = (20,15)) # histogram of all pollutants
    plt.show()


def Plot_Correlations(data):
    sns.heatmap(data,vmin=-1, vmax=1, annot=True, cmap="RdBu", linewidths=1) # heat map of correlations
    plt.show()
######################################################################################################################################################################

Section(" DATASET INFO ")
Dset_info(dataset)
Section("")

# looking at thism I realised the min value in all the columns is -200... we can't have a negative count of any pollutant, the min should be 0

print((dataset == -200).sum()) # number of -200s per column

# since we can's have negative values, we'll treat them as nan values, and we'll fill them with a best guess when we get to buildinng the model
# since this data is time series, we'd most likely use the data from nearby indexes and similar circimstances to find the best fits

# for now, replace all the -200s with nan

Section(" DATA CLEANING ")

dataset.replace(-200,np.nan, inplace=True)
Dset_info(dataset)

# since we are missing alot of values, I'm curious how many rows only have nan values, they are useless to us

Pollutant_Rows = dataset.columns[2:]

Full_Nans = (dataset[Pollutant_Rows].isna()).all(axis=1)

print(f"Entire nan rows: {Full_Nans.sum()}") #31 fully nan rows

dataset = dataset.loc[~Full_Nans] # remove rows with all nans

Section("")

Section(" DATA VISUALISING ")


Plot_histogram(dataset)

corr_matrix = dataset.corr(numeric_only=True)
Plot_Correlations(corr_matrix)

# with the histogram, there seems to be strong correlations between multiple pollutants

Section("")

print(corr_matrix["C6H6(GT)"].sort_values(ascending=False))
