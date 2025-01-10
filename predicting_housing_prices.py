import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from xgboost import XGBRegressor

#Loading the dataset(downloaded from Kaggle)
df = pd.read_csv('house.csv')

print(df.head())

# Splitting into Features and Target
X = df.drop(columns="MEDV")
y = df["MEDV"]

# Log Transformation for Skewed Features
for col in ["CRIM", "DIS", "LSTAT"]:
    X[col] = np.log1p(X[col])

# Optional: Adding Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Gradient Boosting Model with Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0]
}

gbr = GradientBoostingRegressor(random_state=42)
grid_search = RandomizedSearchCV(
    estimator=gbr,
    param_distributions=param_grid,
    n_iter=20,
    scoring='r2',
    cv=5,
    verbose=2,
    random_state=42
)
grid_search.fit(X_train, y_train)

# Best Parameters
print("Best Parameters:", grid_search.best_params_)
best_gbr = grid_search.best_estimator_

# Evaluation on Test Set
y_pred = best_gbr.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Stacking Regressor
stack = StackingRegressor(
    estimators=[
        ('ridge', Ridge(alpha=1.0)),
        ('rf', RandomForestRegressor(n_estimators=200, random_state=42))
    ],
    final_estimator=GradientBoostingRegressor(**grid_search.best_params_, random_state=42)
)
stack.fit(X_train, y_train)

# Evaluate Stacking Model
y_pred_stack = stack.predict(X_test)
print("Stacking R² Score:", r2_score(y_test, y_pred_stack))

# Cross-Validation Comparison
models = {
    'Gradient Boosting': best_gbr,
    'Stacking': stack
}

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, scoring='r2', cv=5)
    print(f'{name}: Average R²: {scores.mean():.4f}')

# Feature Importance Visualization
feature_importances = best_gbr.feature_importances_
features = poly.get_feature_names_out(X.columns)

# Select Top Features
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df.head(10), x='Importance', y='Feature', palette='viridis')
plt.title('Top 10 Feature Importances')
plt.tight_layout()
plt.show()