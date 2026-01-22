import pandas as pd

DATA_PATH = r'c:\Users\HP\Documents\GitHub\Visa-Status-Prediction\data\visa_data_preprocessed.csv'

try:
    print("Reading dataset...")
    df = pd.read_csv(DATA_PATH)
    
    print(f"Total Rows: {len(df)}")
    
    print("\nMean Processing Days for H-1B1 Singapore (Certified Only):")
    certified_sg = df[(df['VISA_CLASS'] == 'H-1B1 Singapore') & (df['CASE_STATUS'] == 'Certified')]
    print(certified_sg['processing_days'].mean())
    print(f"Count: {len(certified_sg)}")

except Exception as e:
    print(f"Error: {e}")
