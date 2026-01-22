import pandas as pd
import os

DATA_PATH = r'c:\Users\HP\Documents\GitHub\Visa-Status-Prediction\data\visa_data_preprocessed.csv'

try:
    df = pd.read_csv(DATA_PATH, nrows=50000)
    print(f"Loaded {len(df)} rows.")
    
    cols_to_check = ['CASE_STATUS', 'VISA_CLASS', 'application_year', 'EMPLOYER_STATE']
    
    for col in cols_to_check:
        if col in df.columns:
            print(f"\nValue Counts for {col}:")
            print(df[col].value_counts().to_string())
            print("-" * 20)
        else:
            print(f"\n{col} not found in data.")
            
    print("\nProcessing Days Stats:")
    print(df['processing_days'].describe())

except Exception as e:
    print(f"Error: {e}")
