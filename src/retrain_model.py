
import pandas as pd
import numpy as np
import warnings
import os
from datetime import datetime
import json
import pickle
import joblib

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# Paths
DATA_PATH = r'c:\Users\HP\Documents\GitHub\Visa-Status-Prediction\data\visa_data_preprocessed.csv'
ARTIFACTS_DIR = r'c:\Users\HP\Documents\GitHub\Visa-Status-Prediction\src'

def retrain():
    print("Loading data...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    # Read the full dataset to ensure we get a random distribution of classes
    # If memory is an issue, we could use chunking, but for <1GB file, full read is usually fine.
    df = pd.read_csv(DATA_PATH)
    print(f"Full Data loaded: {df.shape}")
    
    # Print diversity stats relative to the full dataset
    if 'CASE_STATUS' in df.columns:
         print("CASE_STATUS distribution in full data:")
         print(df['CASE_STATUS'].value_counts())

    # Balanced/Stratified Sampling Strategy
    # The dataset is dominated by H-1B, causing the model to ignore rare classes.
    # We will downsample H-1B but keep ALL minority classes.
    
    print("Applying balanced sampling...")
    majority_class = 'H-1B'
    
    # 1. Split Majority vs Minority
    df_minority = df[df['VISA_CLASS'] != majority_class]
    df_majority = df[df['VISA_CLASS'] == majority_class]
    
    print(f"Majority (H-1B) count: {len(df_majority)}")
    print(f"Minority (Others) count: {len(df_minority)}")
    
    # 2. Downsample Majority
    # We want to force the model to pay equal attention to minority classes.
    # Strategy: Aggressive 1:1 Ratio. Take exactly len(df_minority) samples of majority.
    target_majority_size = len(df_minority)
    
    # Ensure we have at least 10k majority samples to be safe, if minority is tiny
    target_majority_size = max(target_majority_size, 10000)
    
    df_majority_sampled = df_majority.sample(n=target_majority_size, random_state=42)
    print(f"Downsampled Majority to: {len(df_majority_sampled)} (Targeting 1:1 Ratio)")
    
    # 3. Combine and Shuffle
    df = pd.concat([df_majority_sampled, df_minority])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Balanced Data for Training: {df.shape}")
    print("Class distribution in balanced set:")
    print(df['VISA_CLASS'].value_counts())
    
    print(f"Data loaded: {df.shape}")

    selected_features = [
        'VISA_CLASS', 'CASE_STATUS', 'FULL_TIME_POSITION', 'EMPLOYER_STATE', 'WORKSITE_STATE',
        'application_year', 'application_month', 'application_season', 'application_weekday',
        'JOB_TITLE', 'SOC_TITLE', 'TOTAL_WORKER_POSITIONS',
        'WAGE_RATE_OF_PAY_FROM', 'WAGE_UNIT_OF_PAY', 'PREVAILING_WAGE', 'PW_UNIT_OF_PAY',
        'NAICS_CODE', 'H_1B_DEPENDENT', 'WILLFUL_VIOLATOR',
        'processing_days'
    ]

    available_features = [f for f in selected_features if f in df.columns]
    X = df[available_features].copy()
    y = X.pop('processing_days')

    # Handle high cardinality
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    for col in categorical_cols:
        if X[col].nunique() > 50:
            top_categories = X[col].value_counts().head(20).index
            X[col] = X[col].apply(lambda x: x if x in top_categories else 'Other')

    # Preprocessing
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Model - Random Forest (as in the app)
    # Increase estimators and depth to capture subtle signals (like H-1B1 Singapore)
    model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    print("Training model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test RMSE: {rmse}")
    print(f"Test MAE: {mae}")
    print(f"Test R2: {r2}")

    # Saving artifacts
    print("Saving artifacts...")
    
    # Save Model (Pipeline including preprocessor)
    # The app expects:
    # MODEL_PATH = 'visa_processing_model_Random_Forest.pkl' -> just the model or pipeline?
    # App code:
    # self.model = joblib.load(MODEL_PATH)
    # self.preprocessor = joblib.load(PREPROCESSOR_PATH)
    # prediction = self.model.predict(input_df)[0]
    
    # WAIT. In the app: 
    # input_df = pd.DataFrame([input_data])
    # self.model.predict(input_df)
    
    # If self.model is just the regressor, it expects transformed input.
    # If self.model is the pipeline, it expects raw input.
    # The original script `visa_model.py` constructs a pipeline AND saves `best_model` (which is a pipeline).
    # BUT the app code loads `model` AND `preprocessor` separately.
    # And calls `self.model.predict(input_df)`.
    # If `model` is the pipeline, it internally calls preprocessor. 
    # Why does the app load `preprocessor` then? 
    # App `__init__`: loads preprocessor but DOES NOT USE IT in `predict_processing_time`.
    # It just loads it.
    # `predict_processing_time` calls `self.model.predict(input_df)`.
    # So `self.model` MUST be the full Pipeline.
    
    # Therefore, we strictly need to save the Pipeline as the model file.
    
    joblib.dump(pipeline, os.path.join(ARTIFACTS_DIR, 'visa_processing_model_Random_Forest.pkl'))
    
    # Save Preprocessor separately (for compatibility with app structure, even if unused)
    joblib.dump(preprocessor, os.path.join(ARTIFACTS_DIR, 'visa_preprocessor.pkl'))
    
    # Save Features
    with open(os.path.join(ARTIFACTS_DIR, 'visa_features.pkl'), 'wb') as f:
        pickle.dump(available_features, f)
        
    # Save Summary
    # Save Summary
    summary = {
        'model_name': 'Random Forest (Retrained)',
        'test_rmse': float(rmse),
        'test_mae': float(mae),
        'test_r2': float(r2),
        'training_samples': len(X_train),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(ARTIFACTS_DIR, 'model_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("Done.")

if __name__ == "__main__":
    retrain()
