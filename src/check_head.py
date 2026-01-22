import pandas as pd

DATA_PATH = r'c:\Users\HP\Documents\GitHub\Visa-Status-Prediction\data\visa_data_preprocessed.csv'

try:
    print("Reading first 1000 rows...")
    df = pd.read_csv(DATA_PATH, nrows=1000)
    
    print(f"CASE_STATUS counts: {dict(df['CASE_STATUS'].value_counts())}")
    print(f"Year counts: {dict(df['application_year'].value_counts())}")
    print(f"State counts: {dict(df['EMPLOYER_STATE'].value_counts())}")

except Exception as e:
    print(f"Error: {e}")
