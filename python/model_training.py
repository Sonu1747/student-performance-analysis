"""
Model Training for Student Dropout Prediction
This script trains and evaluates machine learning models for dropout prediction.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
import os
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DropoutPredictionModel:
    """Class for training and managing dropout prediction models."""
    
    def __init__(self, model_type: str = 'random_forest'):
        """Initialize the model trainer."""
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.model_metrics = {}
        self.training_history = {}
        
    def prepare_data(self, df: pd.DataFrame, target_column: str = 'dropout_risk') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for model training."""
        logger.info("Preparing data for model training...")
        
        # Select numerical features
        numerical_features = [
            'age', 'avg_score', 'min_score', 'max_score', 'total_exams', 'subjects_count',
            'attendance_rate', 'total_days', 'total_activities', 'total_duration', 'login_sessions',
            'grade_consistency', 'grade_volatility', 'attendance_score', 'avg_session_duration',
            'engagement_intensity', 'combined_risk_score', 'subject_diversity',
            'academic_engagement_score', 'performance_consistency', 'attendance_engagement'
        ]
        
        # Filter existing columns
        available_features = [col for col in numerical_features if col in df.columns]
        
        # Add encoded categorical features
        categorical_encoded = [col for col in df.columns if col.endswith('_encoded')]
        available_features.extend(categorical_encoded)
        
        # Remove any columns with all NaN values
        available_features = [col for col in available_features if not df[col].isna().all()]
        
        # Prepare features and target
        X = df[available_features].fillna(0)
        y = df[target_column]
        
        # Remove rows with missing target
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        self.feature_columns = available_features
        logger.info(f"Data prepared: {X.shape[0]} samples, {len(available_features)} features")
        
        return X, y
    
    def train_random_forest(self, X: pd.DataFrame, y: pd.Series, 
                          test_size: float = 0.2, random_state: int = 42) -> Dict:
        """Train Random Forest model."""
        logger.info("Training Random Forest model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define parameter grid for hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Initialize model
        rf = RandomForestClassifier(random_state=random_state, class_weight='balanced')
        
        # Grid search for best parameters
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Get best model
        self.model = grid_search.best_estimator_
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        metrics['best_params'] = grid_search.best_params_
        
        # Store training history
        self.training_history = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        logger.info("Random Forest model training completed")
        return metrics
    
    def train_gradient_boosting(self, X: pd.DataFrame, y: pd.Series, 
                              test_size: float = 0.2, random_state: int = 42) -> Dict:
        """Train Gradient Boosting model."""
        logger.info("Training Gradient Boosting model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Initialize model
        gb = GradientBoostingClassifier(random_state=random_state)
        
        # Grid search
        grid_search = GridSearchCV(
            gb, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Get best model
        self.model = grid_search.best_estimator_
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        metrics['best_params'] = grid_search.best_params_
        
        # Store training history
        self.training_history = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        logger.info("Gradient Boosting model training completed")
        return metrics
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict:
        """Calculate model performance metrics."""
        metrics = {
            'accuracy': (y_true == y_pred).mean(),
            'precision': classification_report(y_true, y_pred, output_dict=True)['weighted avg']['precision'],
            'recall': classification_report(y_true, y_pred, output_dict=True)['weighted avg']['recall'],
            'f1_score': classification_report(y_true, y_pred, output_dict=True)['weighted avg']['f1-score'],
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_feature_importance(self, top_n: int = 15, save_path: str = None):
        """Plot feature importance."""
        importance_df = self.get_feature_importance()
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df.head(top_n), x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to: {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, save_path: str = None):
        """Plot ROC curve."""
        if not self.training_history:
            raise ValueError("No training history available. Please train the model first.")
        
        y_test = self.training_history['y_test']
        y_pred_proba = self.training_history['y_pred_proba']
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to: {save_path}")
        
        plt.show()
    
    def save_model(self, model_path: str = "models/dropout_prediction_model.pkl"):
        """Save trained model and scaler."""
        if self.model is None:
            raise ValueError("No model to save. Please train the model first.")
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_metrics': self.model_metrics,
            'training_date': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to: {model_path}")
    
    def load_model(self, model_path: str = "models/dropout_prediction_model.pkl"):
        """Load trained model and scaler."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.model_metrics = model_data.get('model_metrics', {})
        
        logger.info(f"Model loaded from: {model_path}")
    
    def train_model(self, df: pd.DataFrame, target_column: str = 'dropout_risk') -> Dict:
        """Train model based on specified type."""
        logger.info(f"Training {self.model_type} model...")
        
        # Prepare data
        X, y = self.prepare_data(df, target_column)
        
        # Train model based on type
        if self.model_type == 'random_forest':
            metrics = self.train_random_forest(X, y)
        elif self.model_type == 'gradient_boosting':
            metrics = self.train_gradient_boosting(X, y)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        self.model_metrics = metrics
        
        # Save model
        self.save_model()
        
        logger.info(f"Model training completed. ROC AUC: {metrics['roc_auc']:.3f}")
        return metrics
    
    def evaluate_model(self) -> Dict:
        """Evaluate trained model."""
        if not self.training_history:
            raise ValueError("No training history available. Please train the model first.")
        
        y_test = self.training_history['y_test']
        y_pred = self.training_history['y_pred']
        y_pred_proba = self.training_history['y_pred_proba']
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Print detailed report
        print("Model Evaluation Results:")
        print("=" * 50)
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1-Score: {metrics['f1_score']:.3f}")
        print(f"ROC AUC: {metrics['roc_auc']:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return metrics

def main():
    """Main function to demonstrate model training."""
    # This would normally load data from feature_engineering.py
    # For demonstration, we'll create sample data
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'student_id': range(1, n_samples + 1),
        'age': np.random.randint(15, 20, n_samples),
        'avg_score': np.random.normal(75, 15, n_samples),
        'min_score': np.random.normal(65, 12, n_samples),
        'max_score': np.random.normal(85, 10, n_samples),
        'total_exams': np.random.randint(3, 8, n_samples),
        'subjects_count': np.random.randint(3, 6, n_samples),
        'attendance_rate': np.random.beta(2, 1, n_samples),
        'total_days': np.random.randint(15, 25, n_samples),
        'total_activities': np.random.poisson(20, n_samples),
        'total_duration': np.random.poisson(1000, n_samples),
        'login_sessions': np.random.poisson(20, n_samples),
        'grade_consistency': np.random.normal(20, 5, n_samples),
        'grade_volatility': np.random.normal(0.2, 0.1, n_samples),
        'attendance_score': np.random.normal(80, 15, n_samples),
        'avg_session_duration': np.random.normal(50, 10, n_samples),
        'engagement_intensity': np.random.normal(1.2, 0.3, n_samples),
        'combined_risk_score': np.random.beta(1, 3, n_samples),
        'subject_diversity': np.random.randint(3, 6, n_samples),
        'academic_engagement_score': np.random.normal(1500, 500, n_samples),
        'performance_consistency': np.random.normal(60, 15, n_samples),
        'attendance_engagement': np.random.normal(16, 5, n_samples)
    })
    
    # Create dropout risk target
    dropout_conditions = (
        (sample_data['avg_score'] < 50) |
        (sample_data['attendance_rate'] < 0.5) |
        (sample_data['total_activities'] < 5) |
        (sample_data['combined_risk_score'] > 0.7)
    )
    sample_data['dropout_risk'] = np.where(dropout_conditions, 1, 0)
    
    # Train Random Forest model
    model_trainer = DropoutPredictionModel(model_type='random_forest')
    metrics = model_trainer.train_model(sample_data)
    
    # Evaluate model
    model_trainer.evaluate_model()
    
    # Plot results
    model_trainer.plot_feature_importance()
    model_trainer.plot_roc_curve()
    
    print(f"\nModel training completed successfully!")
    print(f"Final ROC AUC Score: {metrics['roc_auc']:.3f}")

if __name__ == "__main__":
    main()