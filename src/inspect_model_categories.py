import joblib
import pandas as pd
import numpy as np

# Load the model
try:
    model = joblib.load('visa_processing_model_Random_Forest.pkl')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

try:
    preprocessor = model.named_steps['preprocessor']
    cat_transformer = preprocessor.named_transformers_['cat']
    onehot = cat_transformer.named_steps['onehot']
    
    # We need the feature names to match categories to columns
    # Reconstructing the logic from visa_model.py
    # Ideally we should have saved the column names used for training
    
    with open('categories.txt', 'w') as f:
        f.write("--- Model Categorical Features ---\n")
        for i, cats in enumerate(onehot.categories_):
            f.write(f"Feature Index {i}\n")
            f.write(f"  Valid Categories ({len(cats)}):\n")
            f.write(f"  {cats}\n")
            if 'Other' in cats:
                f.write("  -> Has 'Other' category\n")
            else:
                f.write("  -> NO 'Other' category\n")
            f.write("\n")
    print("Categories written to categories.txt")
            
except Exception as e:
    print(f"Error inspecting model: {e}")
