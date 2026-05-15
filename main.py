#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Main execution script for Diamond Price Prediction"""

import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.svr_model import SVRModel
from src.models.linear_regression import LinearRegressionModel
from src.models.knn_model import KNNModel
from src.utils.metrics import ModelMetrics
from src.utils.visualization import Visualizer


def main():
    """Main function to run all models"""
    
    # Load data
    print("Loading data...")
    loader = DataLoader('data/Diamonds - Regression.csv')
    df = loader.load_data()
    loader.get_basic_info(df)
    
    # Preprocess data
    print("\nPreprocessing data...")
    preprocessor = DataPreprocessor(test_size=0.2, random_state=42)
    df_encoded = preprocessor.encode_categorical(df)
    X, y, feature_names = preprocessor.prepare_features(df_encoded)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    # Initialize metrics and visualizer
    metrics_calc = ModelMetrics()
    viz = Visualizer()
    
    # 1. Linear Regression
    print("\n" + "="*60)
    print("Training Linear Regression Model...")
    lr_model = LinearRegressionModel()
    lr_model.train(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)
    lr_metrics = metrics_calc.calculate_metrics(y_test, y_pred_lr)
    metrics_calc.print_metrics("Multiple Linear Regression", lr_metrics)
    
    # Feature importance
    importance = lr_model.get_feature_importance(feature_names)
    print("\nFeature Coefficients:")
    print(importance)
    
    # 2. KNN Model
    print("\n" + "="*60)
    print("Training KNN Model...")
    knn_model = KNNModel(n_neighbors=5)
    knn_model.train(X_train_scaled, y_train)
    y_pred_knn = knn_model.predict(X_test_scaled)
    knn_metrics = metrics_calc.calculate_metrics(y_test, y_pred_knn)
    metrics_calc.print_metrics("KNN Regressor (k=5)", knn_metrics)
    
    # 3. SVR Model
    print("\n" + "="*60)
    print("Training SVR Model...")
    svr_model = SVRModel(kernel='rbf', C=1.0, epsilon=0.1)
    svr_model.train(X_train_scaled, y_train)
    y_pred_svr = svr_model.predict(X_test_scaled)
    svr_metrics = metrics_calc.calculate_metrics(y_test, y_pred_svr)
    metrics_calc.print_metrics("SVR (Base)", svr_metrics)
    
    # Visualizations for best model (KNN)
    print("\nGenerating visualizations for best model (KNN)...")
    viz.plot_predictions(y_test, y_pred_knn, "KNN Regressor")
    viz.plot_residuals(y_test, y_pred_knn)
    
    # Compare all models
    print("\n" + "="*60)
    print("Model Comparison Summary")
    print("="*60)
    comparison_df = pd.DataFrame({
        'Model': ['Linear Regression', 'KNN (k=5)', 'SVR (Base)'],
        'R2 Score': [lr_metrics['R2'], knn_metrics['R2'], svr_metrics['R2']],
        'RMSE': [lr_metrics['RMSE'], knn_metrics['RMSE'], svr_metrics['RMSE']],
        'MAE': [lr_metrics['MAE'], knn_metrics['MAE'], svr_metrics['MAE']]
    })
    print(comparison_df.to_string(index=False))
    
    return {
        'linear_regression': lr_model,
        'knn': knn_model,
        'svr': svr_model,
        'metrics': comparison_df
    }


if __name__ == "__main__":
    results = main()