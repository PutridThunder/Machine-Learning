import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.impute import SimpleImputer

# we should aim to build multiple models, one with just pollutant readings, then add in temperature, then time, and compare each model's performance

# 1. Select A Problem = CO(GT) levels

# 2. know what tools to use

# 3. select a performance measure

PROJECT_ROOT = Path(__file__).parent 
DATA_PATH = PROJECT_ROOT / "Data" / "AirQualityUCI.xlsx"

dataset = pd.read_excel(DATA_PATH)

dataset.replace(-200,np.nan, inplace=True)

Pollutant_Rows = dataset.columns[2:]
Full_Nans = (dataset[Pollutant_Rows].isna()).all(axis=1)
print(f"Entire nan rows: {Full_Nans.sum()}") #31 fully nan rows

dataset = dataset.loc[~Full_Nans] # remove rows with all nans

# I want to replace all Nan values with the best guess features,
# since there are many empty  indexes, replacing them all with the average
# wouldn't be very helpful to us

# we know there are 31 rows that were fully Nan
# 90% of NMHC(GT) is gone, we should drop it
# for the rest, we are about 17% of it being missing, not too bad

dataset.drop(columns=["NMHC(GT)"], inplace=True)

# we could replace the nan values with the average, I think a better decision is to use neighboring values to create an estimation for the nan values
# say we have a high concentration, followed by many nans then a low, throughout the day it is realistic for it to change overtime

# linear interpolation looks at two points and assumes the gap between them is linear, fair
# another option would be hybrid interpolation, combining linear interpolation with what we know about the data from various times on various days
# for simplification, I will use linear, though it would be interesting on comparing preprocessing strategies and seeing how it affect results

# first lets split the data into testing and training, in testing, we should also remove rows with nan values for CO(GT), we won't test on artificial values

# found out pandas has datetime installed already, no missing values from date or time, so lets combine those columns using datetime too

dataset["Datetime"] = dataset["Date"] + pd.to_timedelta(dataset["Time"].astype(str))

# now that we have this, the date and time columns have no importance

dataset.drop(columns=["Date","Time"], inplace=True)

# now lets split, since its time series and not indexed like housing data
# the best recorded method for time series is chronological splitting, not random splitting
# we'll use the first 80% of the dataset for training, and reserve the last 20% for testing

def chronological_split(df):
    "returns as (training, testing), training being 80% of the dataframe"
    SplitIdx = int(len(df) * 0.8)
    training = df.iloc[:SplitIdx].copy()
    testing = df.iloc[SplitIdx:].copy()

    return training, testing

training_set, testing_set = chronological_split(dataset)

target = "CO(GT)"

features = [
    "PT08.S1(CO)",
    "C6H6(GT)",
    "PT08.S2(NMHC)",
    "NOx(GT)",
    "PT08.S3(NOx)",
    "NO2(GT)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)"
]

training_set.dropna(subset=[target], inplace=True)
testing_set = testing_set.dropna(subset=[target] + features)

# now we'll do linear interpolation, both here means we will look at a before and after point and fill in all in between nans with best guesses

training_set = training_set.interpolate(method="linear", limit_direction="both") 

# time to train the model
# for this project, we're only using pollutant/sensor readings
# we're intentionally leaving out temperature, humidity, and time as a seperate experiment for now

X_train = training_set[features]
y_train = training_set[target]

# the testing data should only contain rows where we actually know the CO concentration
# otherwise we can't evaluate the model

X_test = testing_set[features]
y_test = testing_set[target]

#we'll train on 3 models, linear regression, tree, and forest

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# linear regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

lin_predictions = lin_reg.predict(X_test)

# decision tree
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train, y_train)

tree_predictions = tree_reg.predict(X_test)

# random forest
forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)

forest_reg.fit(X_train, y_train)

forest_predictions = forest_reg.predict(X_test)

# now let's evaluate the models

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def Evaluate_Model(name, y_true, predictions):
    """
    evaluates a regression model using RMSE, MAE, and R²
    name = name of the model,
    y_true is the actual target value,
    predictions is the predicted value
    """

    mse = mean_squared_error(y_true, predictions)
    rmse = np.sqrt(mse)

    mae = mean_absolute_error(y_true, predictions)

    r2 = r2_score(y_true, predictions)

    print(f"\n{name}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")

Evaluate_Model("Linear Regression",y_test,lin_predictions)
Evaluate_Model("Decision Tree",y_test,tree_predictions)
Evaluate_Model("Random Forest",y_test,forest_predictions)
