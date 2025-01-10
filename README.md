# Predicting-Housing-Prices
##Boston Housing Price Prediction with Gradient Boosting
This project explores the use of Gradient Boosting and Stacking Regressors to predict housing prices using the Boston Housing Dataset. The code demonstrates feature engineering, hyperparameter tuning, and ensemble modeling for optimal performance.

###Gradient Boosting Model
Gradient Boosting is a powerful ensemble method that builds a sequence of models, each correcting errors made by the previous ones. It is particularly effective in handling overfitting through regularization.

###Hyperparameter Tuning
We tuned the following hyperparameters using GridSearchCV:
learning_rate
max_depth
n_estimators
subsample

###Files in this Repository
predicting_housing_prices.py: The main Python script that loads the dataset, performs preprocessing, trains the model, and tunes hyperparameters.
house.csv: The dataset used for this project.

##Features
1. Preprocessing:
Log Transformation: Address skewed features for better normalization.
Polynomial Features: Add interaction terms to capture non-linear relationships.
Train-Test Split: Standard 80/20 split for robust evaluation.
2. Modeling:
Gradient Boosting: Tuned with RandomizedSearchCV for best parameters.
Stacking Regressor: Combines Ridge Regression, Random Forest, and Gradient Boosting for enhanced predictions.
3. Evaluation:
R² Score, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) metrics.
Feature Importance visualization for model interpretability.

###Outputs include:
Best Gradient Boosting parameters.
Metrics on test data (R², MAE, RMSE).
Feature importance plot.
Requirements
Python 3.7+
Libraries:
text
Copy code
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
