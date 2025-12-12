# AI Enabled Visa Status Prediction & Processing Time Estimator
## Milestone 1 – Data Collection & Preprocessing

### 📊 Dataset
- **Source**: Combined_LCA_Disclosure_Data_FY2024  
- **Format**: CSV with 97 columns of visa application data  
- **Raw file location**: Google Drive (path specified in code)  
- **Kaggle Reference Link**: [Kaggle Dataset Link](https://www.kaggle.com/datasets/zongaobian/h1b-lca-disclosure-data-2020-2024?select=Combined_LCA_Disclosure_Data_FY2024.csv)


### 🛠️ Steps Performed

#### 1. **Data Loading**
- Mounted Google Drive in Colab environment  
- Loaded the raw CSV file using pandas  
- Initial dataset shape: 97 columns with visa application records

#### 2. **Missing Value Handling**
- Identified columns with missing values  
- **Numerical columns**: Filled missing values with median  
- **Categorical columns**: Filled missing values with mode  
- **Date columns**: Converted to datetime format, handled invalid dates

#### 3. **Date Processing**
- Converted relevant date columns to datetime format:
  - `RECEIVED_DATE` → Application submission date  
  - `DECISION_DATE` → Application decision date  
  - `ORIGINAL_CERT_DATE` → Original certification date  
  - `BEGIN_DATE` → Employment start date  
  - `END_DATE` → Employment end date  

#### 4. **Target Variable Creation**
- Created new target column: `processing_days`  
- Formula: `processing_days = DECISION_DATE - RECEIVED_DATE` (in days)  
- Removed invalid records with negative processing times  
- Processing time statistics:
  - **Mean**: Calculated from data  
  - **Median**: Calculated from data  
  - **Range**: 0 to max realistic processing days  

#### 5. **Data Cleaning**
- Stripped whitespace from all string columns  
- Standardized categorical values  
- Handled inconsistent text formatting  

#### 6. **Feature Engineering**
- Extracted temporal features from `RECEIVED_DATE`:
  - `application_year`  
  - `application_month`  
  - `application_day`  
  - `application_weekday`  
  - `application_season` (Winter, Spring, Summer, Fall)

#### 7. **Exploratory Data Analysis**
Generated visualizations for:  
- Distribution of processing days  
- Top 10 visa classes by frequency  
- Average processing time by visa class  
- Case status distribution  

### 📈 Key Features Identified
From the 97 original columns, key predictive features include:
- `CASE_STATUS`  
- `VISA_CLASS`  
- `JOB_TITLE`  
- `FULL_TIME_POSITION`  
- `EMPLOYER_STATE`  
- `WORKSITE_STATE`  
- `processing_days` (target variable)

### 📁 Outputs
**Processed file saved to**: `/content/drive/MyDrive/visa_data_preprocessed.csv`  
- Cleaned dataset with all original columns  
- Added: `processing_days` + temporal features  
- Ready for feature selection and modeling  

### 🚀 How to Run in Google Colab

1. **Prepare your data:**
   - Upload CSV file to Google Drive  
   - Note file path  

2. **Update the code:**
   ```python
   file_path = '/content/drive/MyDrive/YOUR_FOLDER/YOUR_FILE.csv'
   ```
3. **Access processed data:**
   File located at: /content/drive/MyDrive/visa_data_preprocessed.csv
