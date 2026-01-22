import sys
import os

# Ensure src is in path
sys.path.append(os.getcwd())

try:
    from app import VisaPredictor
except ImportError:
    print("Could not import VisaPredictor from app.py. Make sure you are in src/")
    exit(1)

# Redirect stdout to file
sys.stdout = open('debug_log.txt', 'w')
print("Initializing Predictor...")
predictor = VisaPredictor()

test_inputs = [
    {
        'visa_class': 'H-1B',
        'employer_state': 'CA',
        'worksite_state': 'CA',
        'job_title': 'Software Engineer',
        'wage_from': '120000',
        'wage_unit': 'Year',
        'full_time': 'Y',
        'test_desc': 'Standard H-1B Tech'
    },
    {
         'visa_class': 'H-1B1 Singapore', 
         'employer_state': 'CA',
         'worksite_state': 'CA',
         'job_title': 'Data Analyst',
         'wage_from': '85000',
         'wage_unit': 'Year',
         'full_time': 'Y',
         'test_desc': 'H-1B1 Singapore (Expected ~22 days)'
    },
    {
         'visa_class': 'E-3 Australian', 
         'employer_state': 'NY',
         'worksite_state': 'NY',
         'job_title': 'Manager',
         'wage_from': '150000', 
         'wage_unit': 'Year',
         'full_time': 'Y',
         'test_desc': 'E-3 Australian'
    }
]

print("\n--- Starting Tests ---")
for i, form_data in enumerate(test_inputs):
    print(f"\nTest {i+1}: {form_data}")
    
    
    # Simulate what happens in app.py route
    try:
        input_data = predictor.prepare_input_data(form_data)
        
        # Override for testing
        if 'year_override' in form_data:
            input_data['application_year'] = form_data['year_override']
            print(f"DEBUG: Overriding Year to {input_data['application_year']}")
            
        print(f"Prepared Input Keys: {list(input_data.keys())}")
        # print(f"Prepared Input Data: {input_data}") # Verbose
        
        result = predictor.predict_processing_time(input_data)
        print(f"Result Days: {result.get('processing_days')}")
        print("-" * 30)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
