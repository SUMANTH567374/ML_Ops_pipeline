import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_data(data_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        return df
    except pd.errors.ParserError as e:
        print(f"Error: Failed to parse the CSV file from {data_url}.")
        print(e)
        raise
    except Exception as e:
        print(f"Error: An unexpected error occurred while loading the data.")
        print(e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        # Assuming this preprocessing step is generic and suitable for your dataset
        # Modify as needed based on the actual structure of Salary_Data.csv
        if 'Salary' not in df.columns or 'Years of Experience' not in df.columns:
            raise KeyError("Expected columns 'YearsExperience' and 'Salary' not found.")
        return df  # No specific preprocessing in this example
    except KeyError as e:
        print(f"Error: Missing column {e} in the dataframe.")
        raise
    except Exception as e:
        print(f"Error: An unexpected error occurred during preprocessing.")
        print(e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        data_path = os.path.join(data_path, 'raw')
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
    except Exception as e:
        print(f"Error: An unexpected error occurred while saving the data.")
        print(e)
        raise

def main():
    try:
        # Path to the local Salary_Data.csv file
        data_path = r"C:\Users\LENOVO\Downloads\Salary_Data.csv"
        df = load_data(data_url=data_path)

        # Preprocess the data
        final_df = preprocess_data(df)

        # Split the data into training and testing sets
        train_data, test_data = train_test_split(final_df, test_size=0.2, random_state=42)

        # Save the split data to disk
        save_data(train_data, test_data, data_path='data')

    except Exception as e:
        print(f"Error: {e}")
        print("Failed to complete the data ingestion process.")

if __name__ == '__main__':
    main()

    

