import pickle
import json
import os
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Load the test data (assuming the test data was also scaled previously)
test_scaled_df = pd.read_csv('./data/scaled/test_scaled.csv')

# Extract features (X) and target (y) from the scaled test data
X_test_scaled = test_scaled_df.drop(columns=['Salary'])  # Assuming 'Salary' is the target column
y_test = test_scaled_df['Salary']

# Load the best model from the pickle file
model_path = './models/best_model.pkl'
try:
    with open(model_path, 'rb') as f:
        best_gbr = pickle.load(f)
    print(f"Best model loaded from '{model_path}'")
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    exit()

# Predict on the test data
y_pred = best_gbr.predict(X_test_scaled)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Define file path to save the metrics
metrics_file_path = './metrics/metrics.json'

# Ensure the output directory exists
os.makedirs(os.path.dirname(metrics_file_path), exist_ok=True)

# Create a dictionary to store the metrics
metrics = {
    'best_params': best_gbr.get_params(),
    'mean_squared_error': mse,
    'r2_score': r2
}

# Save the metrics to a JSON file
with open(metrics_file_path, 'w') as f:
    json.dump(metrics, f, indent=4)

print(f"Metrics saved to '{metrics_file_path}'")

