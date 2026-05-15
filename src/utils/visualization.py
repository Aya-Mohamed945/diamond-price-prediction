"""Visualization module"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class Visualizer:
    """Handle all visualization tasks"""
    
    @staticmethod
    def plot_correlation_matrix(df, figsize=(12, 10)):
        """Plot correlation matrix heatmap"""
        plt.figure(figsize=figsize)
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                   fmt='.2f', linewidths=0.5)
        plt.title('Correlation Matrix of Features and Price')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_predictions(y_test, y_pred, model_name):
        """Plot actual vs predicted values"""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 
                'r--', lw=2)
        plt.xlabel('Actual Prices')
        plt.ylabel('Predicted Prices')
        plt.title(f'{model_name}: Actual vs Predicted Prices')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_residuals(y_test, y_pred):
        """Plot residuals distribution"""
        residuals = y_test - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Predicted Values')
        
        # Residuals distribution
        axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Residuals Distribution')
        
        plt.tight_layout()
        plt.show()
