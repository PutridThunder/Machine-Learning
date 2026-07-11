
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import numpy as np


PROJECT_ROOT = Path(__file__).parent 
DATA_PATH = PROJECT_ROOT / "housing.csv"

housing = pd.read_csv(DATA_PATH)

####################################################################################################
# helper functions

def Section(text):
    """Prints a section divider to make terminal output easier to read."""
    print(f"\n================================================================================{text}================================================================================\n")


def Dataset_Info(dataset):
    """Displays basic information about a dataframe."""

    print(f"Shape: {dataset.shape}")
    dataset.info()
    print(dataset.describe())


####################################################################################################
# data visualization

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


def Plot_Scatter(dataset, x_feature, y_feature):
    """Creates a scatter plot between two numerical features."""

    dataset.plot(
        kind="scatter",
        x=x_feature,
        y=y_feature,
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()


def Plot_Boxplots(dataset):
    """Displays boxplots for every numerical feature.

    Useful for identifying outliers.
    """

    dataset.plot(
        kind="box",
        subplots=True,
        figsize=(18, 8),
        layout=(3, 3),
        sharex=False,
        sharey=False
    )

    plt.tight_layout()
    plt.show()


def Plot_Pairplot(dataset, columns):
    """Displays pairwise relationships between selected features."""

    sns.pairplot(dataset[columns])

    plt.show()


####################################################################################################
# correlation helpers

def Feature_Correlations(dataset, target):
    """Returns every feature's correlation with a chosen target feature."""

    corr_matrix = dataset.corr(numeric_only=True)

    correlations = corr_matrix[target].sort_values(ascending=False)

    print(correlations)

    return correlations

Plot_Histograms(housing)
Plot_Geographical_Data(housing)
Plot_Correlation_Heatmap(housing)

corr_matrix = housing.corr(numeric_only=True)
print(corr_matrix)
print(corr_matrix["median_house_value"].sort_values(ascending=False))
