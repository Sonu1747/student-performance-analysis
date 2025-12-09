# Student Performance Monitoring System - How It Works

## Overview
This system monitors student performance and predicts dropout risk using machine learning. Here's how each component works:

## System Architecture Flow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Database      │───▶│  Data Pipeline  │───▶│  Feature Eng.   │───▶│  Model Training │
│                 │    │                 │    │                 │    │                 │
│ • Students      │    │ • Extract       │    │ • Create        │    │ • Train ML      │
│ • Grades        │    │ • Clean         │    │   features      │    │   Model         │
│ • Attendance    │    │ • Merge         │    │ • Encode        │    │ • Evaluate      │
│ • Engagement    │    │ • Validate      │    │ • Scale         │    │ • Save Model    │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
                                                                              │
                                                                              ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Power BI       │◀───│  Predictions    │◀───│  Load Model     │    │   Automation    │
│  Dashboard      │    │                 │    │                 │    │                 │
│                 │    │ • Risk Scores   │    │ • Load .pkl     │    │ • Daily Pulls   │
│ • Visualizations│    │ • Risk Levels   │    │ • Predict       │    │ • Alerts        │
│ • Alerts        │    │ • Alerts        │    │ • Save Results  │    │ • Reports       │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Step-by-Step Explanation

### 1. Data Pipeline (`data_pipeline.py`)

**Purpose:** Extract, clean, and combine data from multiple database tables

**What it does:**
- Connects to SQLite/MySQL database
- Extracts 4 types of data:
  - **Students**: Name, age, gender, class, enrollment date
  - **Grades**: Subject scores, exam dates, terms
  - **Attendance**: Daily attendance records (Present/Absent/Late)
  - **Engagement**: Login activities, session duration, activity types

**Output Example:**
```
student_id | name          | avg_score | attendance_rate | total_activities
-----------|---------------|-----------|-----------------|------------------
1          | John Smith    | 85.5      | 0.95           | 25
2          | Sarah Johnson | 76.5      | 0.85           | 15
3          | Mike Davis    | 65.0      | 0.70           | 8
```

**Result:** `data/processed/student_data_processed.csv`

---

### 2. Feature Engineering (`feature_engineering.py`)

**Purpose:** Create advanced features for machine learning

**What it creates:**
- **Academic Features**: Grade consistency, volatility, performance categories
- **Attendance Features**: Attendance scores, risk indicators
- **Engagement Features**: Activity intensity, session duration averages
- **Combined Features**: Risk scores combining academic + attendance + engagement
- **Categorical Encodings**: Convert text categories to numbers

**Key Features Created:**
- `combined_risk_score`: Weighted combination (Academic 40% + Attendance 30% + Engagement 30%)
- `dropout_risk`: Binary target (1 = at risk, 0 = not at risk)
- `risk_level`: Categories (Low, Medium, High, Critical)

**Output Example:**
```
student_id | avg_score | attendance_rate | combined_risk_score | dropout_risk | risk_level
-----------|-----------|-----------------|---------------------|--------------|------------
1          | 85.5      | 0.95           | 0.15                | 0            | Low
3          | 65.0      | 0.70           | 0.65                | 1            | High
```

---

### 3. Model Training (`model_training.py`)

**Purpose:** Train machine learning model to predict dropout risk

**Model Type:** Random Forest Classifier

**Process:**
1. Split data: 80% training, 20% testing
2. Hyperparameter tuning with GridSearchCV
3. Train model on training data
4. Evaluate on test data
5. Calculate metrics: Accuracy, Precision, Recall, F1-Score, ROC AUC

**Example Output:**
```
Model Performance Metrics:
- Accuracy:  0.875 (87.5%)
- Precision: 0.892
- Recall:    0.834
- F1-Score:  0.862
- ROC AUC:   0.921
```

**Result:** `models/dropout_prediction_model.pkl`

---

### 4. Predictions (`predict.py`)

**Purpose:** Use trained model to predict dropout risk for all students

**What it does:**
1. Load trained model
2. Apply to new student data
3. Generate risk scores (0-1 probability)
4. Categorize into risk levels
5. Identify high-risk students
6. Generate alerts

**Output Example:**
```
Prediction Summary:
- Total Students: 50
- High Risk Students: 8 (16.0%)
- Average Risk Score: 0.342
- Maximum Risk Score: 0.876

Risk Distribution:
- Low:     32 students (64.0%)
- Medium:   8 students (16.0%)
- High:     7 students (14.0%)
- Critical: 3 students (6.0%)

High-Risk Students:
student_id | name        | risk_score | risk_level
-----------|-------------|------------|------------
15         | Student_15  | 0.876      | Critical
23         | Student_23  | 0.743      | Critical
31         | Student_31  | 0.698      | High
```

**Result:** `data/processed/predictions.csv`

---

### 5. Automation (`automation.py`)

**Purpose:** Automate daily/weekly tasks

**Scheduled Tasks:**
- **Daily 6:00 AM**: Pull fresh data from database
- **Daily 8:00 AM**: Run risk assessment and send alerts
- **Weekly Sunday 2:00 AM**: Retrain model with new data
- **Weekly Sunday 3:00 AM**: Generate weekly reports

**Alert System:**
- Email alerts for high-risk students
- Model update notifications
- Error notifications

---

## Example Workflow

### Scenario: Identifying At-Risk Students

1. **Data Collection** (Daily)
   - System pulls latest grades, attendance, and engagement data

2. **Feature Creation** (Automatic)
   - Calculates risk scores for each student
   - Combines academic, attendance, and engagement metrics

3. **Prediction** (Daily)
   - Model predicts dropout probability
   - Students with risk > 60% flagged as high-risk

4. **Alert Generation** (Automatic)
   - Email sent to counselors/admins
   - Lists high-risk students with recommendations

5. **Intervention** (Manual)
   - Staff contacts high-risk students
   - Provides academic support
   - Monitors progress

6. **Model Update** (Weekly)
   - Model retrained with latest data
   - Improved predictions over time

---

## Key Metrics Explained

### Risk Score (0-1)
- **0.0-0.2**: Low Risk - Student is performing well
- **0.2-0.4**: Medium Risk - Monitor closely
- **0.4-0.6**: High Risk - Intervention needed
- **0.6-1.0**: Critical Risk - Immediate action required

### Contributing Factors
1. **Academic Performance (40% weight)**
   - Average scores < 60%
   - Declining grades
   - Failed exams

2. **Attendance (30% weight)**
   - Attendance rate < 70%
   - Frequent absences
   - Pattern of skipping classes

3. **Engagement (30% weight)**
   - Low activity participation
   - Few login sessions
   - Short session durations

---

## Files Generated

1. **data/processed/student_data_processed.csv**
   - Clean, merged student data

2. **models/dropout_prediction_model.pkl**
   - Trained machine learning model

3. **data/processed/predictions.csv**
   - Risk predictions for all students

4. **reports/alerts_YYYYMMDD_HHMMSS.json**
   - High-risk student alerts

5. **logs/automation.log**
   - System activity log

---

## How to Use

### Run Complete Pipeline:
```bash
python run_pipeline.py
```

### Run Individual Components:
```bash
# Data Pipeline
python python/data_pipeline.py

# Feature Engineering
python python/feature_engineering.py

# Model Training
python python/model_training.py

# Make Predictions
python python/predict.py
```

### Start Automation:
```bash
python python/automation.py
```

---

## Expected Outputs

When you run the system, you'll see:

1. **Data Pipeline Output:**
   - Number of records processed
   - Sample of processed data
   - Summary statistics

2. **Feature Engineering Output:**
   - Number of features created
   - Risk distribution
   - Feature importance

3. **Model Training Output:**
   - Training progress
   - Model performance metrics
   - Feature importance rankings

4. **Predictions Output:**
   - Total students analyzed
   - High-risk count and percentage
   - Risk level distribution
   - List of high-risk students
   - Generated alerts

---

This system helps schools proactively identify and support students at risk of dropping out!

