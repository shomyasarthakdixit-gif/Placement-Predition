import os
import sys
import pandas as pd

# Add root project directory to sys.path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH, PROCESSED_DATA_DIR

def clean_data():
    """
    Cleans the raw dataset and saves it to the processed data directory.
    - Fills missing values with median for numerical columns.
    - Drops duplicate rows.
    """
    print(f"Loading raw data from: {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)
    
    # Fill missing values for numerical columns with their median
    missing_cols = ['Workshops', 'AptitudeTestScore', 'SoftSkillsRating', 'CodingTestScore', 'MockInterviewScore']
    for col in missing_cols:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Filled missing values in '{col}' with median: {median_val}")
            
    # Drop duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df.drop_duplicates(inplace=True)
        print(f"Dropped {duplicates} duplicate rows.")
        
    # Ensure processed directory exists
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Save cleaned data
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Cleaned data saved to: {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    clean_data()
