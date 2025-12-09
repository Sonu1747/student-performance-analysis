# Student Performance Monitoring & Dropout Prediction System

A comprehensive system for tracking student performance, predicting dropout risk, and providing actionable insights through automated data processing and interactive dashboards.

## 🎯 Project Overview

This system helps educational institutions:
- **Monitor student performance** across multiple subjects and time periods
- **Predict dropout risk** using machine learning algorithms
- **Track attendance patterns** and engagement metrics
- **Generate automated alerts** for high-risk students
- **Provide interactive dashboards** for data visualization

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   Python ML    │    │   Power BI     │
│                 │    │   Pipeline     │    │   Dashboard    │
│ • Students      │───▶│                 │───▶│                 │
│ • Grades        │    │ • Data Pipeline│    │ • Visualizations│
│ • Attendance    │    │ • Feature Eng. │    │ • Alerts       │
│ • Engagement    │    │ • ML Training  │    │ • Reports      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
student_performance_project/
├── data/
│   ├── raw/                    # Raw data files
│   └── processed/              # Processed data files
├── sql/
│   └── schema.sql             # Database schema
├── python/
│   ├── data_pipeline.py       # Data extraction and cleaning
│   ├── feature_engineering.py # Feature creation and selection
│   ├── model_training.py      # ML model training
│   ├── predict.py            # Prediction and scoring
│   └── automation.py         # Scheduled tasks and alerts
├── powerbi/
│   └── dashboard_guide.md     # Power BI setup guide
├── config/
│   └── config.yaml           # Configuration settings
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8 or higher
- SQLite or MySQL database
- Power BI Desktop (for dashboards)
- Git (for version control)

### 2. Installation

```bash
# Clone the repository
git clone <repository-url>
cd student_performance_project

# Install Python dependencies
pip install -r requirements.txt

# Set up database (SQLite)
sqlite3 student_performance.db < sql/schema.sql
```

### 3. Configuration

Edit `config/config.yaml` to match your environment:

```yaml
database:
  type: "sqlite"  # or "mysql"
  database: "student_performance.db"
  
email:
  smtp_server: "smtp.gmail.com"
  username: "your_email@gmail.com"
  password: "your_app_password"
```

### 4. Run the System

```bash
# Test the installation
python test_scripts.py

# Run data pipeline
python python/data_pipeline.py

# Train the model
python python/model_training.py

# Make predictions
python python/predict.py

# Start automation
python python/automation.py
```

## 🔧 Components

### 1. Data Pipeline (`data_pipeline.py`)

**Purpose**: Extract, clean, and merge data from multiple sources.

**Features**:
- Database connectivity (SQLite/MySQL)
- Data validation and cleaning
- Missing value handling
- Data merging and aggregation

**Usage**:
```python
from data_pipeline import StudentDataPipeline

pipeline = StudentDataPipeline()
processed_data = pipeline.run_pipeline()
```

### 2. Feature Engineering (`feature_engineering.py`)

**Purpose**: Create advanced features for machine learning.

**Features**:
- Academic performance metrics
- Attendance rate calculations
- Engagement intensity measures
- Risk factor combinations

**Usage**:
```python
from feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
features = engineer.run_feature_engineering(data)
```

### 3. Model Training (`model_training.py`)

**Purpose**: Train machine learning models for dropout prediction.

**Features**:
- Random Forest classifier
- Gradient Boosting classifier
- Hyperparameter tuning
- Model evaluation and validation

**Usage**:
```python
from model_training import DropoutPredictionModel

model = DropoutPredictionModel(model_type='random_forest')
metrics = model.train_model(data)
```

### 4. Prediction System (`predict.py`)

**Purpose**: Make predictions on new data and generate alerts.

**Features**:
- Risk score calculation
- Alert generation
- Report export
- Database integration

**Usage**:
```python
from predict import DropoutPredictor

predictor = DropoutPredictor()
predictions = predictor.run_prediction_pipeline()
```

### 5. Automation (`automation.py`)

**Purpose**: Schedule and automate system tasks.

**Features**:
- Daily data pulls
- Weekly model updates
- Email alerts
- Power BI refresh

**Usage**:
```python
from automation import StudentPerformanceAutomation

automation = StudentPerformanceAutomation()
automation.run_automation()
```

## 📊 Power BI Dashboard

### Dashboard Components

1. **Executive Summary**
   - Key performance indicators
   - Risk distribution charts
   - Trend analysis

2. **Student Performance Tracker**
   - Individual student cards
   - Performance metrics
   - Risk level indicators

3. **Attendance Heatmap**
   - Calendar view
   - Color-coded attendance
   - Pattern analysis

4. **Engagement Trends**
   - Activity timelines
   - Engagement metrics
   - Session analysis

5. **Dropout Risk Analysis**
   - Risk score visualization
   - Risk factor breakdown
   - Intervention recommendations

### Setup Instructions

1. Connect to your data source
2. Import the data model
3. Create visualizations as per the guide
4. Set up scheduled refresh
5. Configure alerts

See `powerbi/dashboard_guide.md` for detailed instructions.

## 🔄 Automation

### Scheduled Tasks

- **Daily (6:00 AM)**: Data pull from database
- **Daily (8:00 AM)**: Risk assessment and alerts
- **Weekly (Sunday 2:00 AM)**: Model retraining
- **Weekly (Sunday 3:00 AM)**: Report generation

### Alert System

- **High-risk student alerts**
- **Performance degradation alerts**
- **System error notifications**
- **Model update notifications**

## 📈 Key Features

### Machine Learning
- **Random Forest** for dropout prediction
- **Feature engineering** for enhanced accuracy
- **Cross-validation** for model reliability
- **Hyperparameter tuning** for optimal performance

### Data Processing
- **Automated data cleaning**
- **Missing value imputation**
- **Outlier detection and handling**
- **Data validation and quality checks**

### Visualization
- **Interactive dashboards**
- **Real-time updates**
- **Mobile-responsive design**
- **Custom visualizations**

### Integration
- **Database connectivity**
- **Email notifications**
- **Power BI integration**
- **REST API support** (future)

## 🛠️ Configuration

### Database Settings

```yaml
database:
  type: "sqlite"  # or "mysql"
  host: "localhost"
  database: "student_performance.db"
  user: "admin"
  password: "password"
```

### Email Settings

```yaml
email:
  smtp_server: "smtp.gmail.com"
  smtp_port: 587
  username: "your_email@gmail.com"
  password: "your_app_password"
  to_emails: ["admin@school.edu"]
```

### Automation Settings

```yaml
automation:
  data_pull_frequency: "daily"
  model_update_frequency: "weekly"
  alert_frequency: "daily"
  high_risk_threshold: 0.6
```

## 📋 Usage Examples

### 1. Running a Complete Analysis

```python
# Load and process data
from data_pipeline import StudentDataPipeline
from feature_engineering import FeatureEngineer
from model_training import DropoutPredictionModel

# Data pipeline
pipeline = StudentDataPipeline()
data = pipeline.run_pipeline()

# Feature engineering
engineer = FeatureEngineer()
features = engineer.run_feature_engineering(data)

# Model training
model = DropoutPredictionModel()
metrics = model.train_model(features)

# Make predictions
from predict import DropoutPredictor
predictor = DropoutPredictor()
predictions = predictor.make_predictions(data)
```

### 2. Setting Up Automation

```python
from automation import StudentPerformanceAutomation

# Initialize automation
automation = StudentPerformanceAutomation()

# Run scheduled tasks
automation.run_automation()
```

### 3. Generating Reports

```python
# Generate risk assessment report
predictor = DropoutPredictor()
predictor.load_model()
predictions = predictor.make_predictions(data)

# Export results
predictor.export_predictions("reports/risk_assessment.csv")

# Generate alerts
alerts = predictor.generate_alerts()
```

## 🔍 Monitoring and Maintenance

### Health Checks
- Database connectivity
- Model performance metrics
- Data quality validation
- System resource usage

### Regular Tasks
- **Daily**: Check data refresh status
- **Weekly**: Review model performance
- **Monthly**: Update data model
- **Quarterly**: Security and compliance review

### Troubleshooting

#### Common Issues
1. **Database connection failures**
   - Check connection settings
   - Verify database credentials
   - Ensure database is running

2. **Model training failures**
   - Check data quality
   - Verify feature availability
   - Review error logs

3. **Email alert failures**
   - Verify SMTP settings
   - Check email credentials
   - Review firewall settings

## 📚 Documentation

- **SQL Schema**: `sql/schema.sql`
- **Power BI Guide**: `powerbi/dashboard_guide.md`
- **Configuration**: `config/config.yaml`
- **API Documentation**: Coming soon

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting guide
- Review the documentation
- Contact the development team

## 🔮 Future Enhancements

- **Real-time streaming** data processing
- **Advanced ML models** (Deep Learning)
- **Mobile application** for alerts
- **API endpoints** for integration
- **Multi-language support**
- **Cloud deployment** options

---

**Last Updated**: December 2024  
**Version**: 1.0  
**Author**: Student Performance Monitoring Team
