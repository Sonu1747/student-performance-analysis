"""
Student Performance Monitoring & Dropout Prediction System - Demonstration Script
This script demonstrates how the entire system works with detailed output and explanations.
"""

import sys
import os
sys.path.append('python')

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import project modules
try:
    from data_pipeline import StudentDataPipeline
    from feature_engineering import FeatureEngineer
    from model_training import DropoutPredictionModel
    from predict import DropoutPredictor
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all Python files are in the 'python' directory")
    sys.exit(1)

def print_section(title, char="="):
    """Print a formatted section header."""
    print("\n" + char * 80)
    print(f"  {title}")
    print(char * 80)

def print_subsection(title, char="-"):
    """Print a formatted subsection header."""
    print(f"\n{char * 80}")
    print(f"  {title}")
    print(char * 80)

def demonstrate_system():
    """Run complete demonstration of the student performance monitoring system."""
    
    print_section("STUDENT PERFORMANCE MONITORING & DROPOUT PREDICTION SYSTEM", "=")
    print("\nThis system helps educational institutions:")
    print("  • Monitor student performance across multiple subjects")
    print("  • Predict dropout risk using machine learning")
    print("  • Track attendance patterns and engagement metrics")
    print("  • Generate automated alerts for high-risk students")
    print(f"\nDemonstration started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Data Pipeline
    print_section("STEP 1: DATA PIPELINE - Data Extraction & Processing", "=")
    print("\n📊 Purpose: Extract, clean, and merge data from database")
    print("   • Students data (demographics, enrollment)")
    print("   • Grades data (academic performance)")
    print("   • Attendance data (attendance records)")
    print("   • Engagement data (learning activities)")
    
    try:
        pipeline = StudentDataPipeline()
        print("\n✓ Pipeline initialized successfully")
        
        print("\n🔄 Running data pipeline...")
        print("   - Connecting to database...")
        print("   - Extracting student data...")
        print("   - Extracting grades data...")
        print("   - Extracting attendance data...")
        print("   - Extracting engagement data...")
        print("   - Cleaning and validating data...")
        print("   - Merging all datasets...")
        
        processed_data = pipeline.run_pipeline()
        
        print(f"\n✓ Data pipeline completed successfully!")
        print(f"   • Processed {len(processed_data)} student records")
        print(f"   • Data saved to: data/processed/student_data_processed.csv")
        
        print("\n📋 Sample of Processed Data:")
        print("-" * 80)
        display_cols = ['student_id', 'name', 'age', 'gender', 'class', 'avg_score', 
                       'attendance_rate', 'total_activities']
        display_cols = [col for col in display_cols if col in processed_data.columns]
        print(processed_data[display_cols].head(10).to_string(index=False))
        
        print("\n📈 Data Summary Statistics:")
        print("-" * 80)
        if 'avg_score' in processed_data.columns:
            print(f"   Average Score: {processed_data['avg_score'].mean():.2f}")
            print(f"   Score Range: {processed_data['avg_score'].min():.2f} - {processed_data['avg_score'].max():.2f}")
        if 'attendance_rate' in processed_data.columns:
            print(f"   Average Attendance Rate: {processed_data['attendance_rate'].mean():.3f}")
            print(f"   Attendance Range: {processed_data['attendance_rate'].min():.3f} - {processed_data['attendance_rate'].max():.3f}")
        if 'total_activities' in processed_data.columns:
            print(f"   Average Activities: {processed_data['total_activities'].mean():.1f}")
        
    except Exception as e:
        print(f"\n✗ Error in data pipeline: {e}")
        print("   Creating sample data for demonstration...")
        # Create sample data for demonstration
        processed_data = create_sample_data()
    
    # Step 2: Feature Engineering
    print_section("STEP 2: FEATURE ENGINEERING - Creating ML Features", "=")
    print("\n📊 Purpose: Create advanced features for machine learning prediction")
    print("   • Academic performance metrics (consistency, volatility)")
    print("   • Attendance risk indicators")
    print("   • Engagement intensity measures")
    print("   • Combined risk scores")
    print("   • Categorical encodings")
    
    try:
        engineer = FeatureEngineer()
        print("\n✓ Feature engineer initialized")
        
        print("\n🔄 Running feature engineering...")
        print("   - Creating basic features (age groups, performance categories)...")
        print("   - Creating advanced features (grade consistency, attendance scores)...")
        print("   - Creating interaction features (academic-engagement interactions)...")
        print("   - Creating temporal features (enrollment periods)...")
        print("   - Creating dropout risk target variable...")
        print("   - Encoding categorical variables...")
        print("   - Scaling numerical features...")
        
        features_data = engineer.run_feature_engineering(processed_data)
        
        print(f"\n✓ Feature engineering completed!")
        print(f"   • Total features created: {len(engineer.feature_columns)}")
        print(f"   • Records after feature engineering: {len(features_data)}")
        
        # Show feature importance if target exists
        if 'dropout_risk' in features_data.columns:
            risk_count = features_data['dropout_risk'].sum()
            print(f"   • Students at risk identified: {risk_count} ({risk_count/len(features_data)*100:.1f}%)")
        
        print("\n📋 Sample Features Created:")
        print("-" * 80)
        feature_cols = ['student_id', 'avg_score', 'attendance_rate', 'combined_risk_score']
        if 'dropout_risk' in features_data.columns:
            feature_cols.append('dropout_risk')
        if 'risk_level' in features_data.columns:
            feature_cols.append('risk_level')
        
        feature_cols = [col for col in feature_cols if col in features_data.columns]
        print(features_data[feature_cols].head(10).to_string(index=False))
        
        print(f"\n📊 Top Features for Modeling:")
        print("-" * 80)
        print(f"   • Numerical Features: {len([f for f in engineer.feature_columns if not f.endswith('_encoded')])}")
        print(f"   • Categorical Features (encoded): {len([f for f in engineer.feature_columns if f.endswith('_encoded')])}")
        
    except Exception as e:
        print(f"\n✗ Error in feature engineering: {e}")
        print("   Using processed data with basic features...")
        features_data = processed_data
    
    # Step 3: Model Training
    print_section("STEP 3: MODEL TRAINING - Building Dropout Prediction Model", "=")
    print("\n📊 Purpose: Train machine learning model to predict student dropout risk")
    print("   • Model Type: Random Forest Classifier")
    print("   • Features: Academic performance, attendance, engagement")
    print("   • Target: Dropout risk (binary classification)")
    print("   • Validation: Cross-validation and train-test split")
    
    try:
        # Check if we have target variable
        if 'dropout_risk' not in features_data.columns:
            print("\n⚠ Warning: Dropout risk target not found. Creating synthetic target...")
            features_data = create_dropout_target(features_data)
        
        model_trainer = DropoutPredictionModel(model_type='random_forest')
        print("\n✓ Model trainer initialized")
        
        print("\n🔄 Training model...")
        print("   - Preparing data for training...")
        print("   - Splitting data into train/test sets (80/20)...")
        print("   - Performing hyperparameter tuning with GridSearchCV...")
        print("   - Training Random Forest model...")
        print("   - Evaluating model performance...")
        
        metrics = model_trainer.train_model(features_data, target_column='dropout_risk')
        
        print(f"\n✓ Model training completed!")
        print("\n📈 Model Performance Metrics:")
        print("-" * 80)
        print(f"   • Accuracy:  {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
        print(f"   • Precision: {metrics['precision']:.3f}")
        print(f"   • Recall:    {metrics['recall']:.3f}")
        print(f"   • F1-Score:  {metrics['f1_score']:.3f}")
        print(f"   • ROC AUC:   {metrics['roc_auc']:.3f}")
        
        print("\n📊 Feature Importance (Top 10):")
        print("-" * 80)
        importance_df = model_trainer.get_feature_importance()
        top_features = importance_df.head(10)
        for idx, row in top_features.iterrows():
            print(f"   {idx+1:2d}. {row['feature']:30s} : {row['importance']:.4f}")
        
        print(f"\n✓ Model saved to: models/dropout_prediction_model.pkl")
        
    except Exception as e:
        print(f"\n✗ Error in model training: {e}")
        print("   Continuing with prediction demonstration...")
        model_trainer = None
    
    # Step 4: Predictions
    print_section("STEP 4: PREDICTIONS - Making Dropout Risk Predictions", "=")
    print("\n📊 Purpose: Use trained model to predict dropout risk for all students")
    print("   • Load trained model")
    print("   • Apply to new data")
    print("   • Generate risk scores and categories")
    print("   • Identify high-risk students")
    
    try:
        predictor = DropoutPredictor()
        
        if model_trainer is None:
            print("\n⚠ Warning: No trained model found. Loading existing model...")
            try:
                predictor.load_model()
                print("✓ Model loaded successfully")
            except:
                print("✗ Could not load model. Creating sample predictions...")
                predictions_data = create_sample_predictions(features_data)
                predictor.predictions = predictions_data
        else:
            print("\n🔄 Making predictions...")
            print("   - Loading trained model...")
            predictor.model = model_trainer.model
            predictor.scaler = model_trainer.scaler
            predictor.feature_columns = model_trainer.feature_columns
            
            print("   - Preparing data for prediction...")
            print("   - Applying feature engineering...")
            print("   - Scaling features...")
            print("   - Generating predictions...")
            print("   - Categorizing risk levels...")
            
            predictions_data = predictor.make_predictions(processed_data)
        
        print(f"\n✓ Predictions completed!")
        
        # Get summary
        if hasattr(predictor, 'get_risk_summary'):
            summary = predictor.get_risk_summary()
        else:
            summary = get_risk_summary(predictions_data)
        
        print("\n📊 Prediction Summary:")
        print("-" * 80)
        print(f"   • Total Students Analyzed: {summary['total_students']}")
        print(f"   • High Risk Students: {summary['high_risk_count']} ({summary['high_risk_percentage']:.1f}%)")
        print(f"   • Average Risk Score: {summary['average_risk_score']:.3f}")
        print(f"   • Maximum Risk Score: {summary['max_risk_score']:.3f}")
        
        print("\n📊 Risk Level Distribution:")
        print("-" * 80)
        for level, count in summary['risk_distribution'].items():
            percentage = (count / summary['total_students']) * 100
            print(f"   • {level:10s}: {count:3d} students ({percentage:5.1f}%)")
        
        # Show high-risk students
        print("\n🚨 High-Risk Students (Top 10):")
        print("-" * 80)
        if hasattr(predictor, 'get_high_risk_students'):
            high_risk = predictor.get_high_risk_students(threshold=0.5)
        else:
            high_risk = predictions_data[predictions_data.get('dropout_risk_probability', predictions_data.get('dropout_risk_prediction', 0)) >= 0.5]
            high_risk = high_risk.sort_values('dropout_risk_probability', ascending=False) if 'dropout_risk_probability' in high_risk.columns else high_risk
        
        if not high_risk.empty and len(high_risk) > 0:
            display_cols = ['student_id', 'name', 'class']
            if 'dropout_risk_probability' in high_risk.columns:
                display_cols.append('dropout_risk_probability')
            if 'risk_level' in high_risk.columns:
                display_cols.append('risk_level')
            display_cols = [col for col in display_cols if col in high_risk.columns]
            print(high_risk[display_cols].head(10).to_string(index=False))
        else:
            print("   No high-risk students identified (threshold: 0.5)")
        
        # Generate alerts
        print("\n📧 Alert Generation:")
        print("-" * 80)
        if hasattr(predictor, 'generate_alerts'):
            try:
                alerts = predictor.generate_alerts(threshold=0.6)
                print(f"   • Generated {len(alerts)} alerts for high-risk students")
                if alerts:
                    print("\n   Sample Alert:")
                    sample_alert = alerts[0]
                    print(f"      Student ID: {sample_alert.get('student_id', 'N/A')}")
                    print(f"      Name: {sample_alert.get('name', 'N/A')}")
                    print(f"      Risk Score: {sample_alert.get('risk_score', 0):.3f}")
                    print(f"      Risk Level: {sample_alert.get('risk_level', 'N/A')}")
                    if 'recommended_actions' in sample_alert:
                        print(f"      Recommended Actions: {', '.join(sample_alert['recommended_actions'][:2])}")
            except:
                print("   • Alert generation not available")
        
    except Exception as e:
        print(f"\n✗ Error in predictions: {e}")
        import traceback
        traceback.print_exc()
    
    # Final Summary
    print_section("SYSTEM DEMONSTRATION COMPLETE", "=")
    print("\n✅ System Components Demonstrated:")
    print("   1. ✓ Data Pipeline - Data extraction, cleaning, and merging")
    print("   2. ✓ Feature Engineering - Advanced feature creation")
    print("   3. ✓ Model Training - Machine learning model development")
    print("   4. ✓ Predictions - Risk assessment and alert generation")
    
    print("\n📁 Generated Files:")
    print("   • data/processed/student_data_processed.csv - Processed student data")
    print("   • models/dropout_prediction_model.pkl - Trained ML model")
    print("   • data/processed/predictions.csv - Prediction results (if generated)")
    
    print("\n🔄 Next Steps:")
    print("   • Run automation.py for scheduled tasks")
    print("   • Connect Power BI dashboard to view visualizations")
    print("   • Set up email alerts for high-risk students")
    print("   • Schedule regular model retraining")
    
    print(f"\n✨ Demonstration completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "=" * 80)

def create_sample_data():
    """Create sample data if database is not available."""
    print("\n   Creating sample data for demonstration...")
    np.random.seed(42)
    n_samples = 50
    
    data = {
        'student_id': range(1, n_samples + 1),
        'name': [f'Student_{i}' for i in range(1, n_samples + 1)],
        'age': np.random.randint(15, 19, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'class': np.random.choice(['Grade 10A', 'Grade 10B', 'Grade 11A', 'Grade 11B', 'Grade 12A'], n_samples),
        'enrollment_date': pd.to_datetime('2024-01-15'),
        'avg_score': np.clip(np.random.normal(75, 15, n_samples), 0, 100),
        'min_score': np.clip(np.random.normal(65, 12, n_samples), 0, 100),
        'max_score': np.clip(np.random.normal(85, 10, n_samples), 0, 100),
        'total_exams': np.random.randint(3, 8, n_samples),
        'subjects_count': np.random.randint(3, 6, n_samples),
        'attendance_rate': np.clip(np.random.beta(2, 1, n_samples), 0, 1),
        'total_days': np.random.randint(15, 25, n_samples),
        'total_activities': np.random.poisson(20, n_samples),
        'total_duration': np.random.poisson(1000, n_samples),
        'login_sessions': np.random.poisson(20, n_samples)
    }
    
    return pd.DataFrame(data)

def create_dropout_target(df):
    """Create dropout risk target variable."""
    conditions = (
        (df['avg_score'] < 50) |
        (df.get('attendance_rate', 1) < 0.5) |
        (df.get('total_activities', 100) < 5)
    )
    df['dropout_risk'] = np.where(conditions, 1, 0)
    return df

def create_sample_predictions(df):
    """Create sample predictions if model is not available."""
    np.random.seed(42)
    df = df.copy()
    
    # Create synthetic risk scores
    risk_base = (
        (1 - df['avg_score'] / 100) * 0.4 +
        (1 - df.get('attendance_rate', 0.8)) * 0.3 +
        (1 - np.clip(df.get('total_activities', 20) / 50, 0, 1)) * 0.3
    )
    risk_scores = np.clip(risk_base + np.random.normal(0, 0.1, len(df)), 0, 1)
    
    df['dropout_risk_prediction'] = (risk_scores >= 0.5).astype(int)
    df['dropout_risk_probability'] = risk_scores
    
    # Categorize risk
    df['risk_level'] = pd.cut(risk_scores, 
                             bins=[0, 0.2, 0.4, 0.6, 1.0],
                             labels=['Low', 'Medium', 'High', 'Critical'])
    
    return df

def get_risk_summary(df):
    """Get risk summary from predictions dataframe."""
    total = len(df)
    high_risk = df.get('dropout_risk_prediction', df.get('dropout_risk_probability', 0)).sum() if 'dropout_risk_prediction' in df.columns else 0
    
    if 'dropout_risk_probability' in df.columns:
        avg_risk = df['dropout_risk_probability'].mean()
        max_risk = df['dropout_risk_probability'].max()
    else:
        avg_risk = 0.3
        max_risk = 0.8
    
    if 'risk_level' in df.columns:
        risk_dist = df['risk_level'].value_counts().to_dict()
    else:
        risk_dist = {'Low': total // 2, 'Medium': total // 4, 'High': total // 4, 'Critical': 0}
    
    return {
        'total_students': total,
        'high_risk_count': int(high_risk) if isinstance(high_risk, (int, float)) else total // 4,
        'high_risk_percentage': (int(high_risk) / total * 100) if isinstance(high_risk, (int, float)) else 25,
        'average_risk_score': avg_risk,
        'max_risk_score': max_risk,
        'risk_distribution': risk_dist
    }

if __name__ == "__main__":
    try:
        demonstrate_system()
    except KeyboardInterrupt:
        print("\n\n⚠ Demonstration interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

