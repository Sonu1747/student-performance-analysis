
import pandas as pd
import numpy as np
import sqlite3
import mysql.connector
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple
import yaml
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StudentDataPipeline:
    """Main class for handling student data pipeline operations."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the pipeline with configuration."""
        self.config = self._load_config(config_path)
        self.connection = None
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}. Using default configuration.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Return default configuration if config file is not found."""
        return {
            'database': {
                'type': 'sqlite',
                'host': 'localhost',
                'database': 'student_performance.db',
                'user': 'admin',
                'password': 'password'
            },
            'data_processing': {
                'min_attendance_rate': 0.7,
                'min_engagement_sessions': 5,
                'grade_threshold': 60.0
            }
        }
    
    def connect_database(self):
        """Establish database connection based on configuration."""
        try:
            db_config = self.config['database']
            
            if db_config['type'] == 'sqlite':
                self.connection = sqlite3.connect(db_config['database'])
                logger.info("Connected to SQLite database")
            elif db_config['type'] == 'mysql':
                self.connection = mysql.connector.connect(
                    host=db_config['host'],
                    database=db_config['database'],
                    user=db_config['user'],
                    password=db_config['password']
                )
                logger.info("Connected to MySQL database")
            else:
                raise ValueError(f"Unsupported database type: {db_config['type']}")
                
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            raise
    
    def extract_students_data(self) -> pd.DataFrame:
        """Extract students data from database."""
        query = """
        SELECT student_id, name, age, gender, class, enrollment_date
        FROM students
        """
        return pd.read_sql(query, self.connection)
    
    def extract_grades_data(self) -> pd.DataFrame:
        """Extract grades data from database."""
        query = """
        SELECT student_id, subject, score, term, exam_date
        FROM grades
        """
        return pd.read_sql(query, self.connection)
    
    def extract_attendance_data(self) -> pd.DataFrame:
        """Extract attendance data from database."""
        query = """
        SELECT student_id, date, status, subject
        FROM attendance
        """
        return pd.read_sql(query, self.connection)
    
    def extract_engagement_data(self) -> pd.DataFrame:
        """Extract engagement data from database."""
        query = """
        SELECT student_id, login_time, activity_type, duration_minutes, subject
        FROM engagement
        """
        return pd.read_sql(query, self.connection)
    
    def clean_students_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean students data."""
        logger.info("Cleaning students data...")
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['student_id'])
        
        # Handle missing values
        df['name'] = df['name'].fillna('Unknown')
        df['gender'] = df['gender'].fillna('Unknown')
        df['class'] = df['class'].fillna('Unknown')
        
        # Convert date columns
        df['enrollment_date'] = pd.to_datetime(df['enrollment_date'])
        
        # Validate age
        df = df[(df['age'] >= 10) & (df['age'] <= 25)]
        
        logger.info(f"Cleaned students data: {len(df)} records")
        return df
    
    def clean_grades_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean grades data."""
        logger.info("Cleaning grades data...")
        
        # Remove invalid scores
        df = df[(df['score'] >= 0) & (df['score'] <= 100)]
        
        # Convert date columns
        df['exam_date'] = pd.to_datetime(df['exam_date'])
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['student_id', 'subject', 'term', 'exam_date'])
        
        logger.info(f"Cleaned grades data: {len(df)} records")
        return df
    
    def clean_attendance_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean attendance data."""
        logger.info("Cleaning attendance data...")
        
        # Convert date columns
        df['date'] = pd.to_datetime(df['date'])
        
        # Standardize status values
        df['status'] = df['status'].str.title()
        valid_statuses = ['Present', 'Absent', 'Late']
        df = df[df['status'].isin(valid_statuses)]
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['student_id', 'date', 'subject'])
        
        logger.info(f"Cleaned attendance data: {len(df)} records")
        return df
    
    def clean_engagement_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean engagement data."""
        logger.info("Cleaning engagement data...")
        
        # Convert datetime columns
        df['login_time'] = pd.to_datetime(df['login_time'])
        
        # Remove invalid durations
        df = df[df['duration_minutes'] >= 0]
        
        # Standardize activity types
        df['activity_type'] = df['activity_type'].str.lower()
        valid_activities = ['quiz', 'video', 'assignment', 'forum', 'resource']
        df = df[df['activity_type'].isin(valid_activities)]
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        logger.info(f"Cleaned engagement data: {len(df)} records")
        return df
    
    def merge_all_data(self, students_df: pd.DataFrame, grades_df: pd.DataFrame, 
                      attendance_df: pd.DataFrame, engagement_df: pd.DataFrame) -> pd.DataFrame:
        """Merge all datasets into a comprehensive dataset."""
        logger.info("Merging all datasets...")
        
        # Start with students as base
        merged_df = students_df.copy()
        
        # Add grades summary
        grades_summary = grades_df.groupby('student_id').agg({
            'score': ['mean', 'min', 'max', 'count'],
            'subject': 'nunique'
        }).round(2)
        
        grades_summary.columns = ['avg_score', 'min_score', 'max_score', 'total_exams', 'subjects_count']
        merged_df = merged_df.merge(grades_summary, on='student_id', how='left')
        
        # Add attendance summary
        attendance_summary = attendance_df.groupby('student_id').agg({
            'status': lambda x: (x == 'Present').sum() / len(x) if len(x) > 0 else 0,
            'date': 'count'
        }).round(3)
        
        attendance_summary.columns = ['attendance_rate', 'total_days']
        merged_df = merged_df.merge(attendance_summary, on='student_id', how='left')
        
        # Add engagement summary
        engagement_summary = engagement_df.groupby('student_id').agg({
            'activity_type': 'count',
            'duration_minutes': 'sum',
            'login_time': 'count'
        }).round(2)
        
        engagement_summary.columns = ['total_activities', 'total_duration', 'login_sessions']
        merged_df = merged_df.merge(engagement_summary, on='student_id', how='left')
        
        # Fill missing values
        numeric_columns = ['avg_score', 'min_score', 'max_score', 'total_exams', 
                          'subjects_count', 'attendance_rate', 'total_days',
                          'total_activities', 'total_duration', 'login_sessions']
        
        for col in numeric_columns:
            if col in merged_df.columns:
                merged_df[col] = merged_df[col].fillna(0)
        
        logger.info(f"Merged dataset: {len(merged_df)} records")
        return merged_df
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str = "data/processed/student_data_processed.csv"):
        """Save processed data to CSV file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to: {output_path}")
    
    def run_pipeline(self, output_path: str = "data/processed/student_data_processed.csv"):
        """Run the complete data pipeline."""
        try:
            logger.info("Starting data pipeline...")
            
            # Connect to database
            self.connect_database()
            
            # Extract data
            students_df = self.extract_students_data()
            grades_df = self.extract_grades_data()
            attendance_df = self.extract_attendance_data()
            engagement_df = self.extract_engagement_data()
            
            # Clean data
            students_df = self.clean_students_data(students_df)
            grades_df = self.clean_grades_data(grades_df)
            attendance_df = self.clean_attendance_data(attendance_df)
            engagement_df = self.clean_engagement_data(engagement_df)
            
            # Merge data
            merged_df = self.merge_all_data(students_df, grades_df, attendance_df, engagement_df)
            
            # Save processed data
            self.save_processed_data(merged_df, output_path)
            
            logger.info("Data pipeline completed successfully!")
            return merged_df
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
        finally:
            if self.connection:
                self.connection.close()
                logger.info("Database connection closed")

def main():
    """Main function to run the data pipeline."""
    pipeline = StudentDataPipeline()
    processed_data = pipeline.run_pipeline()
    print(f"Pipeline completed. Processed {len(processed_data)} student records.")
    print("\nSample of processed data:")
    print(processed_data.head())

if __name__ == "__main__":
    main()