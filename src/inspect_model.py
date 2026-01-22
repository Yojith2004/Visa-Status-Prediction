import joblib
import pandas as pd
import numpy as np
import os

MODEL_PATH = r'src/visa_processing_model_Random_Forest.pkl'

def inspect():
    if not os.path.exists(MODEL_PATH):
        print("Model not found.")
        return

    print(f"Loading model from {MODEL_PATH}...")
    pipeline = joblib.load(MODEL_PATH)
    
    # Check Preprocessor steps
    preprocessor = pipeline.named_steps['preprocessor']
    cat_transformer = preprocessor.named_transformers_['cat']
    onehot = cat_transformer.named_steps['onehot']
    
    print("\nCategorical Features (OneHot):")
    if hasattr(onehot, 'categories_'):
        # Feature 0 is VISA_CLASS
        cats = onehot.categories_[0]
        if 'H-1B1 Singapore' in cats:
            print(f"FOUND 'H-1B1 Singapore' in Feature 0!")
        else:
            print("H-1B1 Singapore NOT FOUND in Feature 0")
            print(f"Categories in Feature 0: {cats}")
    
    # Test Prediction Direct
    print("\n--- Direct Prediction Test ---")
    data_singapore = {
        'VISA_CLASS': ['H-1B1 Singapore'],
        'CASE_STATUS': ['Certified'],
        'FULL_TIME_POSITION': ['Y'],
        'EMPLOYER_STATE': ['CA'],
        'WORKSITE_STATE': ['CA'],
        'JOB_TITLE': ['Software Engineer'],
        'SOC_TITLE': ['Other'],
        'TOTAL_WORKER_POSITIONS': [1.0],
        'WAGE_RATE_OF_PAY_FROM': [85000.0],
        'WAGE_UNIT_OF_PAY': ['Year'],
        'PREVAILING_WAGE': [85000.0],
        'PW_UNIT_OF_PAY': ['Year'],
        'NAICS_CODE': ['541511'],
        'H_1B_DEPENDENT': ['N'],
        'WILLFUL_VIOLATOR': ['N'],
        'application_year': [2026],
        'application_month': [1],
        'application_day': [6],
        'application_weekday': [1],
        'application_season': ['Winter']
    }
    
    df = pd.DataFrame(data_singapore)
    pred = pipeline.predict(df)[0]
    print(f"Prediction for H-1B1 Singapore: {pred}")
    
    data_h1b = data_singapore.copy()
    data_h1b['VISA_CLASS'] = ['H-1B']
    df_h1b = pd.DataFrame(data_h1b)
    pred_h1b = pipeline.predict(df_h1b)[0]
    print(f"Prediction for H-1B: {pred_h1b}")

if __name__ == "__main__":
    inspect()
