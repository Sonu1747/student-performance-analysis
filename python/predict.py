"""
Prediction Script for Student Dropout Risk
This script loads trained models and makes predictions on new data.
"""

import pandas as pd
import numpy as np
import joblib
import sqlite3
import mysql.connector
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Any
import os
import yaml
from data_pipeline import StudentDataPipeline
from feature_engineering import FeatureEngineer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DropoutPredictor:
    """Class for making dropout predictions using trained models."""
    
    def __init__(self, model_path: str = "models/dropout_prediction_model.pkl", 
                 config_path: str = "config/config.yaml"):
        """Initialize the predictor."""
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.model_metrics = {}
        self.predictions = None
        
    def load_model(self):
        """Load trained model and scaler."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.model_metrics = model_data.get('model_metrics', {})
            
            logger.info(f"Model loaded successfully from: {self.model_path}")
            logger.info(f"Model features: {len(self.feature_columns)}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def prepare_prediction_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for prediction using the same preprocessing as training."""
        logger.info("Preparing data for prediction...")
        
        # Create feature engineer instance
        engineer = FeatureEngineer()
        
        # Run feature engineering pipeline
        df_processed = engineer.run_feature_engineering(df)
        
        # Select only the features used in training
        available_features = [col for col in self.feature_columns if col in df_processed.columns]
        missing_features = [col for col in self.feature_columns if col not in df_processed.columns]
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Fill missing features with zeros
            for feature in missing_features:
                df_processed[feature] = 0
        
        # Select and order features as in training
        X_pred = df_processed[available_features].fillna(0)
        
        # Add missing features with zeros
        for feature in missing_features:
            X_pred[feature] = 0
        
        # Reorder columns to match training
        X_pred = X_pred[self.feature_columns]
        
        logger.info(f"Prediction data prepared: {X_pred.shape}")
        return X_pred
    
    def make_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model not loaded. Please load the model first.")
        
        logger.info("Making predictions...")
        
        # Prepare data
        X_pred = self.prepare_prediction_data(df)
        
        # Scale features
        X_pred_scaled = self.scaler.transform(X_pred)
        
        # Make predictions
        predictions = self.model.predict(X_pred_scaled)
        prediction_proba = self.model.predict_proba(X_pred_scaled)
        
        # Create results dataframe
        results_df = df.copy()
        results_df['dropout_risk_prediction'] = predictions
        results_df['dropout_risk_probability'] = prediction_proba[:, 1]
        results_df['risk_level'] = self._categorize_risk(prediction_proba[:, 1])
        results_df['prediction_date'] = datetime.now().date()
        
        # Add confidence scores
        results_df['confidence_score'] = np.max(prediction_proba, axis=1)
        
        self.predictions = results_df
        
        logger.info(f"Predictions completed for {len(results_df)} students")
        logger.info(f"High risk students: {sum(predictions)}")
        
        return results_df
    
    def _categorize_risk(self, probabilities: np.ndarray) -> List[str]:
        """Categorize risk levels based on probabilities."""
        risk_levels = []
        for prob in probabilities:
            if prob < 0.2:
                risk_levels.append('Low')
            elif prob < 0.4:
                risk_levels.append('Medium')
            elif prob < 0.6:
                risk_levels.append('High')
            else:
                risk_levels.append('Critical')
        return risk_levels
    
    def get_high_risk_students(self, threshold: float = 0.5) -> pd.DataFrame:
        """Get students with high dropout risk."""
        if self.predictions is None:
            raise ValueError("No predictions available. Please run predictions first.")
        
        high_risk = self.predictions[
            self.predictions['dropout_risk_probability'] >= threshold
        ].sort_values('dropout_risk_probability', ascending=False)
        
        return high_risk
    
    def get_risk_summary(self) -> Dict:
        """Get summary of risk distribution."""
        if self.predictions is None:
            raise ValueError("No predictions available. Please run predictions first.")
        
        summary = {
            'total_students': len(self.predictions),
            'high_risk_count': sum(self.predictions['dropout_risk_prediction']),
            'high_risk_percentage': (sum(self.predictions['dropout_risk_prediction']) / len(self.predictions)) * 100,
            'risk_distribution': self.predictions['risk_level'].value_counts().to_dict(),
            'average_risk_score': self.predictions['dropout_risk_probability'].mean(),
            'max_risk_score': self.predictions['dropout_risk_probability'].max()
        }
        
        return summary
    
    def export_predictions(self, output_path: str = "data/processed/predictions.csv", 
                          format: str = 'csv') -> str:
        """Export predictions to file."""
        if self.predictions is None:
            raise ValueError("No predictions available. Please run predictions first.")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if format.lower() == 'csv':
            self.predictions.to_csv(output_path, index=False)
        elif format.lower() == 'excel':
            self.predictions.to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Predictions exported to: {output_path}")
        return output_path
    
    def save_predictions_to_database(self, table_name: str = 'dropout_predictions'):
        """Save predictions to database."""
        if self.predictions is None:
            raise ValueError("No predictions available. Please run predictions first.")
        
        try:
            # Load configuration
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            db_config = config['database']
            
            # Connect to database
            if db_config['type'] == 'sqlite':
                conn = sqlite3.connect(db_config['database'])
            elif db_config['type'] == 'mysql':
                conn = mysql.connector.connect(
                    host=db_config['host'],
                    database=db_config['database'],
                    user=db_config['user'],
                    password=db_config['password']
                )
            else:
                raise ValueError(f"Unsupported database type: {db_config['type']}")
            
            # Prepare data for database
            db_data = self.predictions[['student_id', 'dropout_risk_prediction', 
                                      'dropout_risk_probability', 'risk_level', 'prediction_date']].copy()
            db_data.columns = ['student_id', 'dropout_risk_score', 'risk_level', 'prediction_date']
            db_data['model_version'] = 'v1.0'
            
            # Insert into database
            db_data.to_sql(table_name, conn, if_exists='append', index=False)
            
            conn.close()
            logger.info(f"Predictions saved to database table: {table_name}")
            
        except Exception as e:
            logger.error(f"Failed to save predictions to database: {str(e)}")
            raise
    
    def generate_alerts(self, threshold: float = 0.6) -> List[Dict]:
        """Generate alerts for high-risk students."""
        if self.predictions is None:
            raise ValueError("No predictions available. Please run predictions first.")
        
        high_risk_students = self.get_high_risk_students(threshold)
        
        alerts = []
        for _, student in high_risk_students.iterrows():
            alert = {
                'student_id': student['student_id'],
                'name': student.get('name', 'Unknown'),
                'class': student.get('class', 'Unknown'),
                'risk_score': student['dropout_risk_probability'],
                'risk_level': student['risk_level'],
                'alert_date': datetime.now().isoformat(),
                'recommended_actions': self._get_recommended_actions(student)
            }
            alerts.append(alert)
        
        logger.info(f"Generated {len(alerts)} alerts for high-risk students")
        return alerts
    
    def _get_recommended_actions(self, student: pd.Series) -> List[str]:
        """Get recommended actions based on student profile."""
        actions = []
        
        if student.get('attendance_rate', 1) < 0.7:
            actions.append("Schedule attendance intervention meeting")
        
        if student.get('avg_score', 100) < 60:
            actions.append("Arrange academic support and tutoring")
        
        if student.get('total_activities', 0) < 10:
            actions.append("Increase engagement through interactive activities")
        
        if student.get('combined_risk_score', 0) > 0.7:
            actions.append("Schedule comprehensive student support meeting")
        
        if not actions:
            actions.append("Monitor student progress closely")
        
        return actions
    
    def run_prediction_pipeline(self, data_source: str = 'database') -> pd.DataFrame:
        """Run complete prediction pipeline."""
        logger.info("Starting prediction pipeline...")
        
        try:
            # Load model
            self.load_model()
            
            # Load data
            if data_source == 'database':
                pipeline = StudentDataPipeline(self.config_path)
                df = pipeline.run_pipeline()
            else:
                # Load from file
                df = pd.read_csv(data_source)
            
            # Make predictions
            predictions = self.make_predictions(df)
            
            # Export results
            self.export_predictions()
            
            # Save to database
            self.save_predictions_to_database()
            
            # Generate alerts
            alerts = self.generate_alerts()
            
            # Print summary
            summary = self.get_risk_summary()
            print("\nPrediction Summary:")
            print("=" * 50)
            print(f"Total Students: {summary['total_students']}")
            print(f"High Risk Students: {summary['high_risk_count']} ({summary['high_risk_percentage']:.1f}%)")
            print(f"Average Risk Score: {summary['average_risk_score']:.3f}")
            print(f"Risk Distribution: {summary['risk_distribution']}")
            print(f"Alerts Generated: {len(alerts)}")
            
            logger.info("Prediction pipeline completed successfully!")
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction pipeline failed: {str(e)}")
            raise

def main():
    """Main function to run predictions."""
    predictor = DropoutPredictor()
    
    # Run prediction pipeline
    predictions = predictor.run_prediction_pipeline()
    
    # Display high-risk students
    high_risk = predictor.get_high_risk_students()
    if not high_risk.empty:
        print("\nHigh-Risk Students:")
        print("=" * 50)
        print(high_risk[['student_id', 'name', 'class', 'dropout_risk_probability', 'risk_level']].head(10))
    
    print(f"\nPrediction pipeline completed successfully!")
    print(f"Results saved to: data/processed/predictions.csv")

if __name__ == "__main__":
    main()
