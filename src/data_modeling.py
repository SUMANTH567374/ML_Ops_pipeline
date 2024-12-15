import pickle
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import os

# Load the scaled data from CSV
train_scaled_df = pd.read_csv('./data/scaled/train_scaled.csv')

# Extract features (X) and target (y)
X_train_scaled = train_scaled_df.drop(columns=['Salary'])  # Assuming 'Salary' is the target column
y_train = train_scaled_df['Salary']

# Define the model
gbr = GradientBoostingRegressor()

# Define parameter grid for tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10]
}

# Perform Grid Search
grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(X_train_scaled, y_train)

# Best model after tuning
best_gbr = grid_search.best_estimator_

# Specify the path where the model will be saved
model_save_path = './models/best_model.pkl'

# Ensure the directory exists
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# Save the best model to a pickle file
try:
    with open(model_save_path, 'wb') as f:
        pickle.dump(best_gbr, f)
    print(f"Best model saved to '{model_save_path}'")
except Exception as e:
    print(f"An error occurred while saving the model: {e}")
