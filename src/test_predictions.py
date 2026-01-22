import requests
import json

url = 'http://127.0.0.1:5000/predict'

test_cases = [
    {
        'visa_class': 'H-1B',
        'employer_state': 'CA',
        'worksite_state': 'CA',
        'job_title': 'Software Engineer',
        'wage_from': '120000',
        'wage_unit': 'Year',
        'case_status': 'Certified',
        'full_time': 'Y'
    },
    {
        'visa_class': 'H-1B',
        'employer_state': 'TX',
        'worksite_state': 'TX',
        'job_title': 'Data Scientist',
        'wage_from': '90000',
        'wage_unit': 'Year',
        'case_status': 'Certified',
         'full_time': 'N'
    },
    {
         'visa_class': 'E-3', # Should map to E-3 Australian
         'employer_state': 'NY',
         'worksite_state': 'NY',
         'job_title': 'Manager',
         'wage_from': '150000',
         'wage_unit': 'Year',
         'case_status': 'Certified',
         'full_time': 'Y'
    }
]

results = []
for i, data in enumerate(test_cases):
    print(f"\n--- Test Case {i+1} ---")
    try:
        response = requests.post(url, data=data)
        if response.status_code == 200:
            res = response.json()
            days = res.get('processing_days')
            results.append(days)
            print(f"Input: {data['job_title']} in {data['worksite_state']}")
            print(f"Result: {days} days")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Exception: {e}")

if len(set(results)) > 1:
    print("\nSUCCESS: Predictions vary based on input!")
else:
    print("\nFAILURE: Predictions are constant.")
    print(f"All results: {results}")
