#assume we've looked through the data, we understand it all

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
#same random seed to ensure training, splitting, and all randomized variables always yields the same results
np.random.seed(42)

#load the data
data = pd.read_csv("housing.csv")

#split data by income groups
data["income_cat"] = pd.cut(data["median_income"], bins = [0., 1.5, 3.0, 4.5, 6., np.inf], labels = [1, 2, 3, 4, 5])

#split data into 2 group, testing and training

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2, random_state=42)

# split bins into testing and training sets
for train_index, test_index in split.split(data, data["income_cat"]): # split.split(what I want to split, What I want to preserve(the bins))

    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
# Separate features and labels

#inputs are all variables except house value

housing = strat_train_set.drop("median_house_value",axis=1)

#outputs estimated house value

housing_labels = strat_train_set["median_house_value"].copy()

# Numerical columns

housing_num = housing.drop("ocean_proximity", axis=1)

##########################################################################################################################################
# Manual method

#imputer used to fill missing values, we use median of the column to put in the best-guess value
imputer = SimpleImputer(strategy="median")

#imputer stores all medians, accessable through imputer.statistics_
imputer.fit(housing_num)

# x is now a numpy array, where by imputer, solves all missing values by the given mean
x = imputer.transform(housing_num)

# turns the numpy array into a dataframe, where columns are named what they are in the csv, 
housing_tr = pd.DataFrame(data=x, columns=housing_num.columns, index=housing_num.index)

# get the entire column of houses of ocean prox
housing_cat = housing[["ocean_proximity"]]

# Use OneHotEncoder to turn labels ("NEAR_BAY", "ISLAND", etc) into a vector [0,1,0,0...], for each of the possible text locations

one_hot_encoder = OneHotEncoder()

# fit_transform does the same thing done with housing_num, only it does both in one step
housing_cat_1hot = one_hot_encoder.fit_transform(housing_cat) 

# now we build a pipeline
# Assume we are given a new house, to sort the data we would have to do all the steps before
# it is more optimal to store all steps in a pipeline, where we can copy, edit, and apply to similar datasets
##########################################################################################################################################
# Numerical preprocessing pipeline

# pipeline method

# exactly what we've done previously, also scales values

# Raw Numerical Data
#         ↓
# SimpleImputer
#         ↓
# StandardScaler
#         ↓
# Processed Numerical Data

# standart scalar does (value - mean) / standard deviation
# because of our data, some numbers are massive and some are tiny
# (median_income ≈ 8, population ≈ 3500, total_rooms ≈ 2600)

# by scaling, it brings values closer together (median_income = 0.8, population = 0.3, total_rooms = -0.7)
# this makes it easier for the model to learn correlation

# although linear regression doesen't need this, (least squares) it is helpful
# matters for other types of regression, neural networks, SVMs, K-Means, and KNN

# better practice to scale

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())] 
    )

num_attribs = list(housing_num.columns)

# Full preprocessing

# handles all text and numerical data, this is a built pipeline we can use instead of manually writing all steps

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), ["ocean_proximity"])
])

# a production quality script would just run housing_prepared = full_pipeline.fit_transform(housing)

processed_data = full_pipeline.fit_transform(housing)

linreg = LinearRegression()

linreg.fit(processed_data, housing_labels)

##########################################################################################################################################
# Predict Performance

# predict how accurate the model is with data it was trained on
# returns the predicted prices of all the houses
predictions = linreg.predict(processed_data)

# now we calculate the error to the actual prices, stored in housing_labels

mse = mean_squared_error(housing_labels, predictions) # mean_squared_error(actual price, predicted price)

# convert to RMSE (Root Mean Square Error)

rmse = np.sqrt(mse)

# this shows that on average, we are $(rmse) off the actual housing price
print(rmse)
##########################################################################################################################################
# now we use cross validation to split the training set into smaller training/validation sets
# This gives a much better estimate of how well the model generalizes.
scores = cross_val_score(
    linreg,
    processed_data,
    housing_labels,
    scoring="neg_mean_squared_error",
    cv=10
)
# linreg: model type to evaluate
# processed_data: the inputs
# housing_lables: the correct outputs
# cv=10: use 10-fold cross validation
# scoring="neg_mean_squared_error": scoring type
# note, typically, lower MSE is better, but sklearn expects higher scores to be better
# so it returns something like -250000000000 instead of 250000000000, its the same error, only ones negative

# now we convert back to RMSE

rmse_scores = np.sqrt(-scores) # -scores = -(negative score) = positive, real score

# now we print
print(rmse_scores)
print("Average:", rmse_scores.mean())
print("Standard deviation:", rmse_scores.std())

# to see error precent:
average_price = housing_labels.mean()

error_percent = (rmse / average_price) * 100

print(f"predictions are {error_percent}% away from true prices")
##########################################################################################################################################
# so far, we are using all variables to make a single linear equation to act as a best fit estimate
# a decision tree uses all other elements to make a prediction

# a decision tree is one of the easiest ML models to understand, it works almost like a flowchart
# instead of an equation, like least squares, it uses a series of yes/no questions
# many if elses to get a final output, where each output is a house price

# during training, the tree tries asking many possible questions
# and selects the questions that best seperates houses into groups with similar prices

# trees can overfit, it can keep asking questions until it isolates a specific house

# a forest is a model that uses multiple trees,
# it is less prone to overfit as it averages out over multiple trees

# each tree is trained on a random sample of the training data (called bootstrapping)
# and a random subset of features at each split
# e.g tree 1 might focus on 3 features of data (income, latitude, population)
# tree 2 migh focus on other features (rooms, ocean proximity, bedrooms)

"""from sklearn.tree import DecisionTreeRegressor"""
from sklearn.ensemble import RandomForestRegressor
##########################################################################################################################################
# trees
# implementing a tree in sckitlearn is almost identical to linear regression

"""
tree = DecisionTreeRegressor(random_state=42)
tree.fit(processed_data, housing_labels)

tree_predictions = tree.predict(processed_data)

tree_mse = mean_squared_error(housing_labels, tree_predictions)
tree_rmse = np.sqrt(tree_mse)

print(f" tree = {tree_rmse}")
"""
# tree rmse may show 0, which may sound amazing, but could also mean it has memorised every answer
# thats why we need cross validation
##########################################################################################################################################
# random forest

forest = RandomForestRegressor(
    random_state=42
)

forest.fit(processed_data, housing_labels)

forest_predictions = forest.predict(processed_data)

forest_mse = mean_squared_error(
    housing_labels,
    forest_predictions
)

forest_rmse = np.sqrt(forest_mse)

print(f" forrest = {forest_rmse}")

##########################################################################################################################################
# cross validation

from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    forest,                                # model we are testing
    processed_data,                        # the inputs
    housing_labels,                        # ghe correct outputs
    scoring="neg_mean_squared_error",      #
    cv=10                                  # number of splits in training
)

rmse_scores = np.sqrt(-scores)

print(" Forest Scores:", rmse_scores)
print("Average RMSE:", rmse_scores.mean())
print("Standard Deviation:", rmse_scores.std())
