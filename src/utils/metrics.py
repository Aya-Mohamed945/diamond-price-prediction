"""Metrics calculation module"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelMetrics:
    """Calculate and display model performance metrics"""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """Calculate all regression metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        }
    
    @staticmethod
    def print_metrics(model_name: str, metrics: dict):
        """Print metrics in formatted way"""
        print(f"\n{'='*50}")
        print(f"== {model_name} Performance ==")
        print(f"{'='*50}")
        print(f"MAE:  ${metrics['MAE']:,.2f}")
        print(f"MSE:  ${metrics['MSE']:,.2f}")
        print(f"RMSE: ${metrics['RMSE']:,.2f}")
        print(f"R2 Score: {metrics['R2']:.4f}")
        print(f"{'='*50}\n")