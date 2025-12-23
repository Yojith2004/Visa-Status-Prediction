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

## Milestone 2 – Exploratory Data Analysis (EDA)
### 🎯 Objectives
- Visualize processing time distributions across visa types and regions

- Identify trends based on seasons, workload patterns, and applicant origin

- Generate feature importance insights for predictive modeling

### 📊 Analysis Sections
#### 1. **Processing Time Distribution Analysis**
- Overall Distribution: Histogram and box plot of processing days

- By Visa Class: Average processing time across different visa types

- By Case Status: Processing time variation by application outcome

- By Employment Type: Full-time vs part-time processing differences

- Statistical Analysis: Skewness, kurtosis, and descriptive statistics

#### 2. **Regional Analysis**
- Employer State Analysis: Processing time patterns across U.S. states

- Worksite State Analysis: Geographic distribution of processing times

- Volume vs Processing Time Correlation: Relationship between application volume and processing speed

- Top/Bottom Performing States: Identification of fastest and slowest processing regions

- Geographic Heatmaps: Visual representation of state-wise processing times

#### 3. **Seasonal Trend Analysis**
- Monthly Trends: Processing time patterns throughout the year

- Seasonal Variations: Winter, Spring, Summer, Fall comparisons

- Application Volume Seasonality: Busiest and slowest application periods

- Year-over-Year Comparisons: Temporal evolution of processing times

- Heatmap Visualizations: Month vs Year processing time matrix

#### 4. **Workload and Processing Center Analysis**
- Weekly Workload Patterns: Application volume trends by week

- Workload vs Processing Time Correlation: Impact of application volume on processing speed

- Industry Analysis (NAICS_CODE): Processing time variations across industries

- Job Title Analysis (SOC_TITLE): Occupation-specific processing patterns

- Temporal Workload Distribution: Peak and off-peak application periods

#### 5. **Feature Importance Analysis**
- Correlation Analysis: Feature relationships with processing days

- Top Predictive Features: Identification of most influential variables

- Correlation Heatmaps: Visual matrix of feature relationships

- Feature Ranking: Ordered list of features by predictive power

- Categorical Feature Impact: Analysis of text variables on processing times

#### 6. **Statistical Insights**
- Processing Time Statistics: Median, mean, and distribution analysis

- Seasonal Performance: Fastest vs slowest processing seasons

- Visa Class Performance: Processing time variations by visa type

- Regional Performance: State-wise processing efficiency

- Case Status Impact: Processing time differences by application outcome

### 📈 Key Visualizations Generated
- Distribution Plots: Processing time histograms and box plots

- Bar Charts: Visa class, case status, and employment type comparisons

- Scatter Plots: Workload vs processing time correlations

- Heatmaps: Geographic and temporal processing patterns

- Line Charts: Monthly and seasonal trend analysis

- Correlation Matrices: Feature relationship visualizations

### 🔍 Key Findings
- Seasonal Patterns: Identification of fastest/slowest processing seasons

- Geographic Variations: State-wise processing time differences

- Visa Type Impact: Significant variations across visa categories

- Workload Effects: Correlation between application volume and processing time

- Feature Importance: Most predictive variables for modeling

### 📁 EDA Outputs
- Statistical Summary: Descriptive statistics and insights

- Visualization Gallery: 5 comprehensive analysis figures

- Feature Correlation Matrix: Relationship analysis for modeling

- Insights Report: Automated generation of key findings