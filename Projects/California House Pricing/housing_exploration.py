import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).parent 
DATA_PATH = PROJECT_ROOT / "housing.csv"

housing = pd.read_csv(DATA_PATH)

####################################################################################################
# Functions
def Dataset_Info(dataset):
    """Displays basic information about a dataframe."""

    print(f"Shape: {dataset.shape}")
    dataset.info()
    print(dataset.describe())
  
def Plot_Histograms(dataset, bins=50):
    """Displays a histogram for every numerical feature in the dataset."""

    dataset.hist(bins=bins, figsize=(20, 15))
    plt.tight_layout()
    plt.show()

def Plot_Geographical_Data(dataset):
    """Plots the California housing locations.

    Point size represents population.
    Colour represents median house value.
    """

    dataset.plot(
        kind="scatter",
        x="longitude",
        y="latitude",
        alpha=0.4,
        s=dataset["population"] / 100,
        label="Population",
        figsize=(10, 7),
        c="median_house_value",
        cmap=plt.get_cmap("jet"),
        colorbar=True,
        sharex=False
    )

    plt.legend()
    plt.show()

def Plot_Correlation_Heatmap(dataset):
    """Displays a correlation heatmap for every numerical feature."""

    corr_matrix = dataset.corr(numeric_only=True)

    plt.figure(figsize=(12, 10))

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        square=True
    )

    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.show()

    return corr_matrix

####################################################################################################

Plot_Histograms(housing)
Plot_Geographical_Data(housing)
Plot_Correlation_Heatmap(housing)
Dataset_Info(housing)
