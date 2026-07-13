
# California Housing Price Prediction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-scikit--learn-orange)
![Libraries](https://img.shields.io/badge/Libraries-Pandas%20%7C%20NumPy%20%7C%20Matplotlib%20%7C%20scikit--learn-green)

An end-to-end supervised machine learning project that predicts California housing prices using demographic and geographic census data. The project implements a complete machine learning workflow—from exploratory data analysis and preprocessing to model training, evaluation, and comparison using scikit-learn.

The repository is organized into two stages:

- **housing_exploration.py** — Data loading, cleaning, visualization, and exploratory data analysis (EDA)
- **housing_model.py** — Data preprocessing, machine learning pipeline, model training, and evaluation

---

## Project Design & Implementation

This project follows a complete machine learning pipeline similar to what would be used in a real-world regression problem.

### Data Exploration (`housing_exploration.py`)

The exploration script focuses on understanding the dataset before any models are built.

It includes:

- Loading and inspecting the California Housing dataset
- Exploring feature distributions
- Identifying missing values
- Computing descriptive statistics
- Visualizing relationships between variables
- Investigating correlations with housing prices
- Understanding which features may be useful for prediction

This stage is essential because machine learning models are only as good as the data they receive.

---

### Machine Learning Pipeline (`housing_model.py`)

The second script implements the complete preprocessing and machine learning workflow.

### Data Preparation

- Stratified train/test split using income categories
- Separation of features and labels
- Handling missing numerical values using **SimpleImputer**
- Encoding categorical features using **OneHotEncoder**
- Feature scaling using **StandardScaler**
- Automated preprocessing using **Pipeline** and **ColumnTransformer**

---

### Model Training

The project compares multiple supervised learning algorithms:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

Each model is trained using the same processed dataset for a fair comparison.

---

### Model Evaluation

Model performance is evaluated using:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- 10-Fold Cross Validation

Cross validation provides a more reliable estimate of model performance by evaluating each model across multiple training/validation splits rather than relying on a single training error.

---

## Machine Learning Workflow

```text
Load Dataset
      │
      ▼
Explore & Visualize Data
      │
      ▼
Stratified Train/Test Split
      │
      ▼
Data Preprocessing
 • Missing Values
 • Encoding
 • Scaling
      │
      ▼
Feature Pipeline
      │
      ▼
Train Models
 • Linear Regression
 • Decision Tree
 • Random Forest
      │
      ▼
Evaluate Models
 • RMSE
 • Cross Validation
```

---

## Motivation

I built this project to gain a deeper understanding of the complete supervised machine learning workflow rather than simply training a model.

The primary objective was to understand every stage of the pipeline, including:

- Preparing real-world datasets
- Handling missing values
- Encoding categorical variables
- Building reusable preprocessing pipelines
- Training multiple regression models
- Evaluating and comparing model performance using cross validation

This project also served as practical experience with the core tools provided by **scikit-learn**, while reinforcing the theory behind regression models and machine learning pipelines.

---

## Learning Resources

This project was developed while studying **Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow (2nd Edition)** by **Aurélien Géron**.

Rather than simply reproducing the examples from the book, the goal of this project was to build the complete machine learning pipeline from scratch while fully understanding each stage of the workflow.

Throughout the implementation, I focused on understanding and documenting:

- Why stratified sampling is used for train/test splitting
- How missing values are handled with `SimpleImputer`
- Why categorical variables require one-hot encoding
- The purpose of feature scaling with `StandardScaler`
- Building reusable preprocessing pipelines using `Pipeline` and `ColumnTransformer`
- Training and comparing multiple regression models
- Evaluating model performance using RMSE and 10-fold cross validation

The code contains extensive comments explaining both **what each component does** and **why it is used**, making the project a learning reference as well as a working machine learning implementation.

While the overall workflow follows the concepts presented in Géron's book, the implementation has been rewritten, reorganized into separate exploration and modeling scripts, and extensively documented as part of my own learning process.

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- scikit-learn

---

## Current Status

- Complete exploratory data analysis
- Complete preprocessing pipeline
- Multiple regression models implemented
- Cross validation for model evaluation
- Modular project structure separating visualization and machine learning
