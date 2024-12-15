import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

# Fetch the data from the processed data
train_data = pd.read_csv('./data/processed/train_processed.csv')
test_data = pd.read_csv('./data/processed/test_processed.csv')

# Check the first few rows of the data to verify it loaded correctly
print("Train Data Sample:")
print(train_data.head())
print("Test Data Sample:")
print(test_data.head())

# Check for missing values
print("Missing values in train data:", train_data.isnull().sum())
print("Missing values in test data:", test_data.isnull().sum())

# Assuming 'Salary' column exists
X_train = train_data.drop(columns=['Salary'])
y_train = train_data['Salary']

X_test = test_data.drop(columns=['Salary'])
y_test = test_data['Salary']

# Scaling the features
scaler = StandardScaler()

# Fit the scaler on the training data and transform
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data using the same scaler
X_test_scaled = scaler.transform(X_test)

# Print shapes to confirm
print("X_train_scaled shape:", X_train_scaled.shape)
print("X_test_scaled shape:", X_test_scaled.shape)

# Create DataFrames from the scaled features
train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
train_scaled_df['Salary'] = y_train  # Adding target variable

test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
test_scaled_df['Salary'] = y_test  # Adding target variable

# Check the data before saving
print("Train Data Sample after Scaling:")
print(train_scaled_df.head())
print("Test Data Sample after Scaling:")
print(test_scaled_df.head())

# Ensure output directory exists
scaled_data_path = os.path.join("data", "scaled")
os.makedirs(scaled_data_path, exist_ok=True)

# Save the scaled data to CSV
train_scaled_df.to_csv(os.path.join(scaled_data_path, "train_scaled.csv"), index=False)
test_scaled_df.to_csv(os.path.join(scaled_data_path, "test_scaled.csv"), index=False)

# Verify files are saved
print("Saved files:", os.listdir(scaled_data_path))


