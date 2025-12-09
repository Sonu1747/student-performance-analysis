"""
Automation Script for Student Performance Monitoring
This script handles scheduled data pulls, model updates, and alerts.
"""

import schedule
import time
import logging
import smtplib
import sqlite3
import mysql.connector
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import pandas as pd
import numpy as np
import os
import yaml
import json
from typing import Dict, List, Any
import requests
from data_pipeline import StudentDataPipeline
from feature_engineering import FeatureEngineer
from model_training import DropoutPredictionModel
from predict import DropoutPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/automation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StudentPerformanceAutomation:
    """Class for automating student performance monitoring tasks."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the automation system."""
        self.config_path = config_path
        self.config = self._load_config()
        self.setup_directories()
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}. Using default configuration.")
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
            'email': {
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'username': 'your_email@gmail.com',
                'password': 'your_password',
                'from_email': 'your_email@gmail.com',
                'to_emails': ['admin@school.edu', 'counselor@school.edu']
            },
            'automation': {
                'data_pull_frequency': 'daily',
                'model_update_frequency': 'weekly',
                'alert_frequency': 'daily',
                'high_risk_threshold': 0.6
            },
            'powerbi': {
                'gateway_url': 'https://your-gateway-url.com',
                'dataset_id': 'your-dataset-id',
                'refresh_key': 'your-refresh-key'
            }
        }
    
    def setup_directories(self):
        """Create necessary directories."""
        directories = ['logs', 'models', 'data/raw', 'data/processed', 'reports']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def daily_data_pull(self):
        """Pull fresh data from database daily."""
        logger.info("Starting daily data pull...")
        
        try:
            # Run data pipeline
            pipeline = StudentDataPipeline(self.config_path)
            processed_data = pipeline.run_pipeline()
            
            # Save timestamp
            with open('logs/last_data_pull.txt', 'w') as f:
                f.write(datetime.now().isoformat())
            
            logger.info(f"Daily data pull completed. Processed {len(processed_data)} records.")
            
        except Exception as e:
            logger.error(f"Daily data pull failed: {str(e)}")
            self.send_error_notification("Daily Data Pull Failed", str(e))
    
    def weekly_model_update(self):
        """Update model with new data weekly."""
        logger.info("Starting weekly model update...")
        
        try:
            # Load fresh data
            pipeline = StudentDataPipeline(self.config_path)
            df = pipeline.run_pipeline()
            
            # Run feature engineering
            engineer = FeatureEngineer()
            df_features = engineer.run_feature_engineering(df)
            
            # Train new model
            model_trainer = DropoutPredictionModel(model_type='random_forest')
            metrics = model_trainer.train_model(df_features)
            
            # Save model with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f"models/dropout_model_{timestamp}.pkl"
            model_trainer.save_model(model_path)
            
            # Update latest model link
            os.symlink(model_path, "models/dropout_prediction_model.pkl")
            
            logger.info(f"Weekly model update completed. ROC AUC: {metrics['roc_auc']:.3f}")
            
            # Send update notification
            self.send_model_update_notification(metrics)
            
        except Exception as e:
            logger.error(f"Weekly model update failed: {str(e)}")
            self.send_error_notification("Weekly Model Update Failed", str(e))
    
    def daily_risk_assessment(self):
        """Run daily risk assessment and generate alerts."""
        logger.info("Starting daily risk assessment...")
        
        try:
            # Load latest model
            predictor = DropoutPredictor()
            predictor.load_model()
            
            # Get fresh data
            pipeline = StudentDataPipeline(self.config_path)
            df = pipeline.run_pipeline()
            
            # Make predictions
            predictions = predictor.make_predictions(df)
            
            # Generate alerts
            alerts = predictor.generate_alerts(self.config['automation']['high_risk_threshold'])
            
            if alerts:
                # Send alert notifications
                self.send_risk_alerts(alerts)
                
                # Save alerts to file
                self.save_alerts_to_file(alerts)
            
            # Update Power BI dataset
            self.refresh_powerbi_dataset()
            
            logger.info(f"Daily risk assessment completed. Generated {len(alerts)} alerts.")
            
        except Exception as e:
            logger.error(f"Daily risk assessment failed: {str(e)}")
            self.send_error_notification("Daily Risk Assessment Failed", str(e))
    
    def send_risk_alerts(self, alerts: List[Dict]):
        """Send email alerts for high-risk students."""
        if not alerts:
            return
        
        try:
            # Create email content
            subject = f"URGENT: {len(alerts)} High-Risk Students Identified"
            
            html_content = self._create_alert_email_content(alerts)
            
            # Send email
            self._send_email(subject, html_content, is_html=True)
            
            logger.info(f"Risk alerts sent for {len(alerts)} students")
            
        except Exception as e:
            logger.error(f"Failed to send risk alerts: {str(e)}")
    
    def _create_alert_email_content(self, alerts: List[Dict]) -> str:
        """Create HTML content for alert email."""
        html = """
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; }
                .alert { background-color: #ffebee; border-left: 4px solid #f44336; padding: 10px; margin: 10px 0; }
                .student { background-color: #f5f5f5; padding: 8px; margin: 5px 0; border-radius: 4px; }
                .high-risk { background-color: #ffcdd2; }
                .critical-risk { background-color: #ffab91; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h2>🚨 High-Risk Student Alert</h2>
            <p>Date: {date}</p>
            <p>Total High-Risk Students: {count}</p>
            
            <h3>Student Details:</h3>
            <table>
                <tr>
                    <th>Student ID</th>
                    <th>Name</th>
                    <th>Class</th>
                    <th>Risk Score</th>
                    <th>Risk Level</th>
                    <th>Recommended Actions</th>
                </tr>
        """.format(
            date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            count=len(alerts)
        )
        
        for alert in alerts:
            risk_class = "critical-risk" if alert['risk_level'] == 'Critical' else "high-risk"
            html += f"""
                <tr class="{risk_class}">
                    <td>{alert['student_id']}</td>
                    <td>{alert['name']}</td>
                    <td>{alert['class']}</td>
                    <td>{alert['risk_score']:.3f}</td>
                    <td>{alert['risk_level']}</td>
                    <td>{', '.join(alert['recommended_actions'])}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h3>Immediate Actions Required:</h3>
            <ul>
                <li>Contact high-risk students and their families</li>
                <li>Schedule intervention meetings</li>
                <li>Assign academic support resources</li>
                <li>Monitor progress closely</li>
            </ul>
            
            <p><strong>Note:</strong> This is an automated alert. Please review the Power BI dashboard for detailed analysis.</p>
        </body>
        </html>
        """
        
        return html
    
    def send_model_update_notification(self, metrics: Dict):
        """Send notification about model update."""
        subject = "Model Update Completed - Student Performance Monitoring"
        
        content = f"""
        Model Update Summary:
        - Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        - ROC AUC Score: {metrics['roc_auc']:.3f}
        - Accuracy: {metrics['accuracy']:.3f}
        - Precision: {metrics['precision']:.3f}
        - Recall: {metrics['recall']:.3f}
        - F1-Score: {metrics['f1_score']:.3f}
        
        The model has been successfully updated and is ready for predictions.
        """
        
        self._send_email(subject, content)
    
    def send_error_notification(self, error_type: str, error_message: str):
        """Send error notification."""
        subject = f"ERROR: {error_type}"
        
        content = f"""
        Error Type: {error_type}
        Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        Error Message: {error_message}
        
        Please check the logs for more details.
        """
        
        self._send_email(subject, content)
    
    def _send_email(self, subject: str, content: str, is_html: bool = False):
        """Send email notification."""
        try:
            email_config = self.config['email']
            
            msg = MIMEMultipart('alternative')
            msg['From'] = email_config['from_email']
            msg['To'] = ', '.join(email_config['to_emails'])
            msg['Subject'] = subject
            
            if is_html:
                msg.attach(MIMEText(content, 'html'))
            else:
                msg.attach(MIMEText(content, 'plain'))
            
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email sent: {subject}")
            
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
    
    def save_alerts_to_file(self, alerts: List[Dict]):
        """Save alerts to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reports/alerts_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(alerts, f, indent=2, default=str)
        
        logger.info(f"Alerts saved to: {filename}")
    
    def refresh_powerbi_dataset(self):
        """Refresh Power BI dataset via REST API."""
        try:
            powerbi_config = self.config['powerbi']
            
            # Power BI REST API endpoint
            url = f"{powerbi_config['gateway_url']}/datasets/{powerbi_config['dataset_id']}/refreshes"
            
            headers = {
                'Authorization': f"Bearer {powerbi_config['refresh_key']}",
                'Content-Type': 'application/json'
            }
            
            data = {
                'notifyOption': 'NoNotification'
            }
            
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 202:
                logger.info("Power BI dataset refresh initiated successfully")
            else:
                logger.warning(f"Power BI refresh failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Power BI refresh failed: {str(e)}")
    
    def generate_weekly_report(self):
        """Generate weekly performance report."""
        logger.info("Generating weekly report...")
        
        try:
            # Load data
            pipeline = StudentDataPipeline(self.config_path)
            df = pipeline.run_pipeline()
            
            # Run predictions
            predictor = DropoutPredictor()
            predictor.load_model()
            predictions = predictor.make_predictions(df)
            
            # Generate report
            report = {
                'report_date': datetime.now().isoformat(),
                'total_students': len(predictions),
                'high_risk_students': sum(predictions['dropout_risk_prediction']),
                'risk_distribution': predictions['risk_level'].value_counts().to_dict(),
                'average_risk_score': predictions['dropout_risk_probability'].mean(),
                'top_risk_factors': self._analyze_risk_factors(predictions)
            }
            
            # Save report
            timestamp = datetime.now().strftime("%Y%m%d")
            filename = f"reports/weekly_report_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Weekly report generated: {filename}")
            
        except Exception as e:
            logger.error(f"Weekly report generation failed: {str(e)}")
    
    def _analyze_risk_factors(self, predictions: pd.DataFrame) -> Dict:
        """Analyze top risk factors."""
        high_risk = predictions[predictions['dropout_risk_prediction'] == 1]
        
        factors = {
            'low_attendance': sum(high_risk['attendance_rate'] < 0.7) / len(high_risk) * 100,
            'poor_grades': sum(high_risk['avg_score'] < 60) / len(high_risk) * 100,
            'low_engagement': sum(high_risk['total_activities'] < 10) / len(high_risk) * 100
        }
        
        return factors
    
    def setup_schedule(self):
        """Set up automated scheduling."""
        logger.info("Setting up automation schedule...")
        
        # Daily tasks
        schedule.every().day.at("06:00").do(self.daily_data_pull)
        schedule.every().day.at("08:00").do(self.daily_risk_assessment)
        
        # Weekly tasks
        schedule.every().sunday.at("02:00").do(self.weekly_model_update)
        schedule.every().sunday.at("03:00").do(self.generate_weekly_report)
        
        logger.info("Automation schedule configured:")
        logger.info("- Daily data pull: 06:00")
        logger.info("- Daily risk assessment: 08:00")
        logger.info("- Weekly model update: Sunday 02:00")
        logger.info("- Weekly report: Sunday 03:00")
    
    def run_automation(self):
        """Run the automation scheduler."""
        logger.info("Starting automation scheduler...")
        
        self.setup_schedule()
        
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                logger.info("Automation stopped by user")
                break
            except Exception as e:
                logger.error(f"Automation error: {str(e)}")
                time.sleep(300)  # Wait 5 minutes before retrying

def main():
    """Main function to run automation."""
    automation = StudentPerformanceAutomation()
    
    print("Student Performance Monitoring Automation")
    print("=" * 50)
    print("Starting automation scheduler...")
    print("Press Ctrl+C to stop")
    
    try:
        automation.run_automation()
    except KeyboardInterrupt:
        print("\nAutomation stopped by user")
    except Exception as e:
        print(f"Automation failed: {str(e)}")

if __name__ == "__main__":
    main()