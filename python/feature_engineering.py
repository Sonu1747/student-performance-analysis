"""
Feature Engineering for Student Performance Analysis
This script creates advanced features for dropout prediction modeling.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Class for creating and engineering features for student performance analysis."""
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        
    def create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic features from raw data."""
        logger.info("Creating basic features...")
        
        df_features = df.copy()
        
        # Age-based features
        df_features['age_group'] = pd.cut(df_features['age'], 
                                        bins=[0, 15, 17, 19, 25], 
                                        labels=['Young', 'Teen', 'Young Adult', 'Adult'])
        
        # Academic performance features
        df_features['performance_category'] = pd.cut(df_features['avg_score'], 
                                                   bins=[0, 60, 70, 80, 100], 
                                                   labels=['Poor', 'Below Average', 'Average', 'Good'])
        
        # Attendance performance
        df_features['attendance_category'] = pd.cut(df_features['attendance_rate'], 
                                                   bins=[0, 0.6, 0.8, 0.9, 1.0], 
                                                   labels=['Poor', 'Below Average', 'Good', 'Excellent'])
        
        # Engagement performance
        df_features['engagement_category'] = pd.cut(df_features['total_activities'], 
                                                   bins=[0, 5, 15, 30, 1000], 
                                                   labels=['Low', 'Moderate', 'High', 'Very High'])
        
        logger.info("Basic features created successfully")
        return df_features
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features for better prediction."""
        logger.info("Creating advanced features...")
        
        df_features = df.copy()
        
        # Academic consistency features
        df_features['grade_consistency'] = df_features['max_score'] - df_features['min_score']
        df_features['grade_volatility'] = df_features['grade_consistency'] / df_features['avg_score']
        
        # Attendance trend features
        df_features['attendance_score'] = df_features['attendance_rate'] * 100
        df_features['attendance_risk'] = np.where(df_features['attendance_rate'] < 0.7, 1, 0)
        
        # Engagement intensity features
        df_features['avg_session_duration'] = df_features['total_duration'] / df_features['login_sessions']
        df_features['engagement_intensity'] = df_features['total_activities'] / df_features['login_sessions']
        
        # Risk indicators
        df_features['academic_risk'] = np.where(df_features['avg_score'] < 60, 1, 0)
        df_features['engagement_risk'] = np.where(df_features['total_activities'] < 10, 1, 0)
        
        # Combined risk score
        df_features['combined_risk_score'] = (
            df_features['academic_risk'] * 0.4 +
            df_features['attendance_risk'] * 0.3 +
            df_features['engagement_risk'] * 0.3
        )
        
        # Subject diversity
        df_features['subject_diversity'] = df_features['subjects_count']
        
        # Time-based features (if enrollment_date is available)
        if 'enrollment_date' in df_features.columns:
            df_features['enrollment_date'] = pd.to_datetime(df_features['enrollment_date'])
            df_features['days_enrolled'] = (datetime.now() - df_features['enrollment_date']).dt.days
            df_features['enrollment_period'] = pd.cut(df_features['days_enrolled'], 
                                                     bins=[0, 30, 90, 180, 365, 1000], 
                                                     labels=['New', 'Recent', 'Established', 'Long-term', 'Veteran'])
        
        logger.info("Advanced features created successfully")
        return df_features
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different variables."""
        logger.info("Creating interaction features...")
        
        df_features = df.copy()
        
        # Academic-Engagement interactions
        df_features['academic_engagement_score'] = df_features['avg_score'] * df_features['total_activities']
        df_features['performance_consistency'] = df_features['avg_score'] * (1 - df_features['grade_volatility'])
        
        # Attendance-Engagement interactions
        df_features['attendance_engagement'] = df_features['attendance_rate'] * df_features['total_activities']
        
        # Gender-based performance (for bias analysis)
        df_features['gender_performance_gap'] = df_features.groupby('gender')['avg_score'].transform('mean')
        
        # Class-based features
        if 'class' in df_features.columns:
            df_features['class_performance_avg'] = df_features.groupby('class')['avg_score'].transform('mean')
            df_features['class_attendance_avg'] = df_features.groupby('class')['attendance_rate'].transform('mean')
        
        logger.info("Interaction features created successfully")
        return df_features
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features for time-series analysis."""
        logger.info("Creating temporal features...")
        
        df_features = df.copy()
        
        # Recent performance indicators (last 30 days simulation)
        df_features['recent_performance'] = df_features['avg_score'] * np.random.uniform(0.8, 1.2, len(df_features))
        df_features['performance_trend'] = np.random.choice(['Improving', 'Declining', 'Stable'], len(df_features))
        
        # Seasonal features
        current_month = datetime.now().month
        df_features['academic_period'] = np.where(
            current_month in [9, 10, 11, 12], 'First Semester',
            np.where(current_month in [1, 2, 3, 4], 'Second Semester', 'Summer')
        )
        
        logger.info("Temporal features created successfully")
        return df_features
    
    def create_dropout_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create dropout target variable based on multiple criteria."""
        logger.info("Creating dropout target variable...")
        
        df_features = df.copy()
        
        # Define dropout criteria
        dropout_conditions = (
            (df_features['avg_score'] < 50) |  # Very poor academic performance
            (df_features['attendance_rate'] < 0.5) |  # Very poor attendance
            (df_features['total_activities'] < 5) |  # Very low engagement
            (df_features['combined_risk_score'] > 0.7)  # High combined risk
        )
        
        df_features['dropout_risk'] = np.where(dropout_conditions, 1, 0)
        
        # Create risk levels
        df_features['risk_level'] = pd.cut(
            df_features['combined_risk_score'], 
            bins=[0, 0.2, 0.4, 0.6, 1.0], 
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        logger.info(f"Dropout target created. {df_features['dropout_risk'].sum()} students at risk out of {len(df_features)}")
        return df_features
    
    def prepare_features_for_modeling(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare features for machine learning modeling."""
        logger.info("Preparing features for modeling...")
        
        df_model = df.copy()
        
        # Select numerical features
        numerical_features = [
            'age', 'avg_score', 'min_score', 'max_score', 'total_exams', 'subjects_count',
            'attendance_rate', 'total_days', 'total_activities', 'total_duration', 'login_sessions',
            'grade_consistency', 'grade_volatility', 'attendance_score', 'avg_session_duration',
            'engagement_intensity', 'combined_risk_score', 'subject_diversity',
            'academic_engagement_score', 'performance_consistency', 'attendance_engagement'
        ]
        
        # Filter existing columns
        numerical_features = [col for col in numerical_features if col in df_model.columns]
        
        # Handle categorical features
        categorical_features = ['gender', 'class', 'age_group', 'performance_category', 
                              'attendance_category', 'engagement_category']
        categorical_features = [col for col in categorical_features if col in df_model.columns]
        
        # Encode categorical variables
        for col in categorical_features:
            if col in df_model.columns:
                le = LabelEncoder()
                df_model[f'{col}_encoded'] = le.fit_transform(df_model[col].astype(str))
                self.label_encoders[col] = le
        
        # Create final feature set
        feature_columns = numerical_features + [f'{col}_encoded' for col in categorical_features]
        feature_columns = [col for col in feature_columns if col in df_model.columns]
        
        # Remove rows with missing values in features
        df_model = df_model.dropna(subset=feature_columns)
        
        # Scale numerical features
        if numerical_features:
            df_model[numerical_features] = self.scaler.fit_transform(df_model[numerical_features])
        
        self.feature_columns = feature_columns
        logger.info(f"Features prepared for modeling: {len(feature_columns)} features")
        
        return df_model, feature_columns
    
    def get_feature_importance(self, df: pd.DataFrame, target_column: str = 'dropout_risk') -> pd.DataFrame:
        """Get feature importance using statistical tests."""
        logger.info("Calculating feature importance...")
        
        # Prepare features
        df_model, feature_columns = self.prepare_features_for_modeling(df)
        
        # Select features and target
        X = df_model[feature_columns]
        y = df_model[target_column]
        
        # Use SelectKBest for feature selection
        selector = SelectKBest(score_func=f_classif, k='all')
        X_selected = selector.fit_transform(X, y)
        
        # Get feature scores
        feature_scores = pd.DataFrame({
            'feature': feature_columns,
            'score': selector.scores_,
            'p_value': selector.pvalues_
        }).sort_values('score', ascending=False)
        
        logger.info("Feature importance calculated successfully")
        return feature_scores
    
    def run_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run complete feature engineering pipeline."""
        logger.info("Starting feature engineering pipeline...")
        
        # Create all features
        df_features = self.create_basic_features(df)
        df_features = self.create_advanced_features(df_features)
        df_features = self.create_interaction_features(df_features)
        df_features = self.create_temporal_features(df_features)
        df_features = self.create_dropout_target(df_features)
        
        # Prepare for modeling
        df_model, feature_columns = self.prepare_features_for_modeling(df_features)
        
        logger.info(f"Feature engineering completed. Final dataset: {len(df_model)} records, {len(feature_columns)} features")
        
        return df_model

def main():
    """Main function to demonstrate feature engineering."""
    # Load sample data (this would normally come from data_pipeline.py)
    sample_data = pd.DataFrame({
        'student_id': [1, 2, 3, 4, 5],
        'age': [16, 17, 16, 17, 18],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'class': ['Grade 10A', 'Grade 11B', 'Grade 10A', 'Grade 11B', 'Grade 12A'],
        'avg_score': [85.5, 76.5, 65.0, 88.0, 92.0],
        'min_score': [78.0, 70.0, 60.0, 82.0, 88.0],
        'max_score': [92.0, 85.0, 70.0, 95.0, 96.0],
        'total_exams': [3, 3, 3, 3, 3],
        'subjects_count': [3, 3, 3, 3, 3],
        'attendance_rate': [0.95, 0.85, 0.70, 0.90, 0.98],
        'total_days': [20, 18, 15, 19, 21],
        'total_activities': [25, 15, 8, 30, 35],
        'total_duration': [1200, 800, 400, 1500, 1200],
        'login_sessions': [25, 15, 8, 30, 35]
    })
    
    # Run feature engineering
    engineer = FeatureEngineer()
    processed_data = engineer.run_feature_engineering(sample_data)
    
    print("Feature engineering completed!")
    print(f"Dataset shape: {processed_data.shape}")
    print(f"Features: {engineer.feature_columns}")
    print("\nSample of processed data:")
    print(processed_data[['student_id', 'dropout_risk', 'risk_level', 'combined_risk_score']].head())

if __name__ == "__main__":
    main()
