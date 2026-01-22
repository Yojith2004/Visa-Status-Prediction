import streamlit as st
import os
import json
import joblib
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Visa Processing Time Predictor (v1.1 DEBUG)",
    page_icon="📅",
    layout="wide"
)

# DEBUG: List files deeply
st.error(f"DEBUG INFO - Current Directory: {os.getcwd()}")
st.error(f"DEBUG INFO - Script Directory: {os.path.dirname(os.path.abspath(__file__))}")
try:
    st.error(f"DEBUG INFO - Files in src: {os.listdir(os.path.dirname(os.path.abspath(__file__)))}")
except Exception as e:
    st.error(f"DEBUG INFO - Error listing dir: {e}")

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class VisaPredictor:
    def __init__(self):
        """Initialize the prediction system"""
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Define paths relative to the script directory
        self.MODEL_PATH = os.path.join(current_dir, 'visa_model_compressed.joblib')
        self.PREPROCESSOR_PATH = os.path.join(current_dir, 'visa_preprocessor.pkl')
        self.FEATURES_PATH = os.path.join(current_dir, 'visa_features.pkl')
        self.SUMMARY_PATH = os.path.join(current_dir, 'model_summary.json')
        
        # Load artifacts
        self._load_artifacts()
        
        # Define allowed values
        self.allowed_values = {
            'VISA_CLASS': ['H-1B', 'H-2A', 'H-2B', 'H-3', 'E-3', 'Other'],
            'CASE_STATUS': ['Certified', 'Denied', 'Withdrawn', 'Certified-Withdrawn'],
            'FULL_TIME_POSITION': ['Y', 'N'],
            'EMPLOYER_STATE': ['CA', 'NY', 'TX', 'NJ', 'FL', 'IL', 'WA', 'MA', 'GA', 'PA', 'Other'],
            'WORKSITE_STATE': ['CA', 'NY', 'TX', 'NJ', 'FL', 'IL', 'WA', 'MA', 'GA', 'PA', 'Other'],
            'WAGE_UNIT_OF_PAY': ['Year', 'Month', 'Week', 'Hour', 'Bi-Weekly'],
            'PW_UNIT_OF_PAY': ['Year', 'Month', 'Week', 'Hour', 'Bi-Weekly'],
            'H_1B_DEPENDENT': ['Y', 'N', 'Unknown'],
            'WILLFUL_VIOLATOR': ['Y', 'N', 'Unknown'],
            'application_season': ['Winter', 'Spring', 'Summer', 'Fall']
        }

    def _load_artifacts(self):
        # Load model
        if os.path.exists(self.MODEL_PATH):
            self.model = joblib.load(self.MODEL_PATH)
        else:
            files_in_dir = os.listdir(os.path.dirname(self.MODEL_PATH))
            st.error(f"Model file not found: {self.MODEL_PATH}")
            st.error(f"Files in directory: {files_in_dir}")
            self.model = None
        
        # Load preprocessor
        if os.path.exists(self.PREPROCESSOR_PATH):
            self.preprocessor = joblib.load(self.PREPROCESSOR_PATH)
        else:
            self.preprocessor = None
        
        # Load categories/features logic... 
        # (Simplified for Streamlit: we assume init is successful if files exist)
        
        # Introspect model categories if possible
        try:
            if hasattr(self.model, 'named_steps'):
                preprocessor = self.model.named_steps['preprocessor']
                cat_transformer = preprocessor.named_transformers_['cat']
                onehot = cat_transformer.named_steps['onehot']
                self.model_categories = onehot.categories_
            else:
                self.model_categories = []
        except:
            self.model_categories = []

    def prepare_input_data(self, form_data):
        """Prepare input data for prediction"""
        current_date = datetime.now()
        
        input_data = {
            'VISA_CLASS': form_data.get('visa_class', 'H-1B'),
            'CASE_STATUS': 'Certified', # Default as we are predicting for successful cases usually
            'FULL_TIME_POSITION': form_data.get('full_time', 'Y'),
            'EMPLOYER_STATE': form_data.get('employer_state', 'CA'),
            'WORKSITE_STATE': form_data.get('worksite_state', 'CA'),
            'JOB_TITLE': form_data.get('job_title', 'Software Developer'),
            'SOC_TITLE': form_data.get('soc_title', 'Software Developers, Applications'),
            'TOTAL_WORKER_POSITIONS': float(form_data.get('worker_positions', 1)),
            'WAGE_RATE_OF_PAY_FROM': float(form_data.get('wage_from', 80000)),
            'WAGE_UNIT_OF_PAY': form_data.get('wage_unit', 'Year'),
            'PREVAILING_WAGE': float(form_data.get('prevailing_wage', 85000)),
            'PW_UNIT_OF_PAY': form_data.get('pw_unit', 'Year'),
            'NAICS_CODE': form_data.get('naics_code', '541511'),
            'H_1B_DEPENDENT': form_data.get('h1b_dependent', 'N'),
            'WILLFUL_VIOLATOR': form_data.get('willful_violator', 'N'),
            'application_year': current_date.year,
            'application_month': current_date.month,
            'application_day': current_date.day,
            'application_weekday': current_date.weekday(),
            'application_season': self._get_season(current_date.month)
        }
        
        # Apply mappings
        return self._validate_and_map_inputs(input_data)

    def _validate_and_map_inputs(self, input_data):
        # Mappings logic from app.py
        bool_map = {'Y': 'Yes', 'N': 'No', 'y': 'Yes', 'n': 'No'}
        for field in ['H_1B_DEPENDENT', 'WILLFUL_VIOLATOR']:
            if input_data.get(field) in bool_map:
                input_data[field] = bool_map[input_data[field]]
                
        visa_map = {'E-3': 'E-3 Australian', 'H-1B1': 'H-1B1 Chile'}
        if input_data.get('VISA_CLASS') in visa_map:
            input_data['VISA_CLASS'] = visa_map[input_data['VISA_CLASS']]

        # Dynamic category mapping
        cat_feature_order = [
            'VISA_CLASS', 'CASE_STATUS', 'FULL_TIME_POSITION', 
            'EMPLOYER_STATE', 'WORKSITE_STATE', 'application_season',
            'JOB_TITLE', 'SOC_TITLE', 'WAGE_UNIT_OF_PAY', 
            'PW_UNIT_OF_PAY', 'H_1B_DEPENDENT', 'WILLFUL_VIOLATOR'
        ]
        
        if self.model_categories:
            for i, feature_name in enumerate(cat_feature_order):
                if i >= len(self.model_categories): break
                if feature_name in input_data:
                    current = input_data[feature_name]
                    valid = self.model_categories[i]
                    
                    if current in valid: continue
                    if isinstance(current, str) and current.upper() in valid:
                        input_data[feature_name] = current.upper()
                        continue
                    if 'Other' in valid:
                        input_data[feature_name] = 'Other'
                        
        return input_data

    def _get_season(self, month):
        if month in [12, 1, 2]: return 'Winter'
        elif month in [3, 4, 5]: return 'Spring'
        elif month in [6, 7, 8]: return 'Summer'
        else: return 'Fall'

    def predict(self, input_data):
        try:
            df = pd.DataFrame([input_data])
            prediction = self.model.predict(df)[0]
            prediction = max(1, min(365, float(prediction)))
            return prediction
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None

@st.cache_resource
def get_predictor():
    return VisaPredictor()

def main():
    st.title("Visa Processing Time Predictor 🇺🇸")
    st.markdown("Estimate the processing time for US Visa applications based on historical data.")
    
    predictor = get_predictor()
    
    if predictor.model is None:
        st.error("Failed to load model. Please ensure model files are present.")
        return

    # Input Form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Application Details")
            visa_class = st.selectbox("Visa Class", predictor.allowed_values['VISA_CLASS'])
            job_title = st.text_input("Job Title", "Software Developer")
            soc_title = st.text_input("SOC Title", "Software Developers, Applications")
            naics_code = st.text_input("NAICS Code", "541511")
            full_time = st.selectbox("Full Time Position", predictor.allowed_values['FULL_TIME_POSITION'])
            
        with col2:
            st.subheader("Employer Details")
            employer_state = st.selectbox("Employer State", predictor.allowed_values['EMPLOYER_STATE'])
            worksite_state = st.selectbox("Worksite State", predictor.allowed_values['WORKSITE_STATE'])
            prevailing_wage = st.number_input("Prevailing Wage ($)", min_value=0.0, value=85000.0)
            wage_from = st.number_input("Wage Offered ($)", min_value=0.0, value=90000.0)
            wage_unit = st.selectbox("Wage Unit", predictor.allowed_values['WAGE_UNIT_OF_PAY'])
            
        st.subheader("Compliance Info")
        c1, c2 = st.columns(2)
        with c1:
            h1b_dep = st.selectbox("H-1B Dependent?", predictor.allowed_values['H_1B_DEPENDENT'])
        with c2:
            willful_viol = st.selectbox("Willful Violator?", predictor.allowed_values['WILLFUL_VIOLATOR'])
            
        submitted = st.form_submit_button("Predict Processing Time", type="primary")
        
    if submitted:
        input_data = {
            'visa_class': visa_class,
            'full_time': full_time,
            'employer_state': employer_state,
            'worksite_state': worksite_state,
            'job_title': job_title,
            'soc_title': soc_title,
            'workers': 1, # Default
            'wage_from': wage_from,
            'wage_unit': wage_unit,
            'prevailing_wage': prevailing_wage,
            'pw_unit': wage_unit, # Assume same unit for simplicity or add another field
            'naics_code': naics_code,
            'h1b_dependent': h1b_dep,
            'willful_violator': willful_viol
        }
        
        # Prepare and Predict
        prepared_data = predictor.prepare_input_data(input_data)
        days = predictor.predict(prepared_data)
        
        if days:
            st.success("Prediction Complete!")
            
            # Display metrics
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Estimated Days", f"{days:.1f}")
            with m2:
                st.metric("Weeks", f"{days/7:.1f}")
            with m3:
                st.metric("Months", f"{days/30:.1f}")
                
            # Interpretation
            if days < 30:
                st.info("🚀 Fast Processing Time")
            elif days < 90:
                st.warning("⏳ Average Processing Time")
            else:
                st.error("🐢 Slow Processing Time")

if __name__ == "__main__":
    main()
