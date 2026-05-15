"""Training script with hyperparameter tuning"""

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.svr_model import SVRModel
from src.models.knn_model import KNNModel
from src.utils.metrics import ModelMetrics
import joblib
import os


def train_with_tuning():
    """Train models with hyperparameter optimization"""
    
    # Load and preprocess data
    loader = DataLoader('data/Diamonds - Regression.csv')
    df = loader.load_data()
    
    preprocessor = DataPreprocessor(test_size=0.2, random_state=42)
    df_encoded = preprocessor.encode_categorical(df)
    X, y, _ = preprocessor.prepare_features(df_encoded)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    metrics = ModelMetrics()
    
    # KNN Hyperparameter Tuning
    print("Performing KNN Hyperparameter Tuning...")
    knn_tuned = KNNModel()
    knn_grid = knn_tuned.hyperparameter_tuning(X_train_scaled, y_train)
    y_pred_knn = knn_tuned.predict(X_test_scaled)
    knn_metrics = metrics.calculate_metrics(y_test, y_pred_knn)
    metrics.print_metrics("KNN (Tuned)", knn_metrics)
    
    # SVR Hyperparameter Tuning
    print("\nPerforming SVR Hyperparameter Tuning...")
    svr_tuned = SVRModel()
    svr_grid = svr_tuned.hyperparameter_tuning(X_train_scaled, y_train)
    y_pred_svr = svr_tuned.predict(X_test_scaled)
    svr_metrics = metrics.calculate_metrics(y_test, y_pred_svr)
    metrics.print_metrics("SVR (Tuned)", svr_metrics)
    
    # Save models
    os.makedirs('models', exist_ok=True)
    joblib.dump(knn_tuned.model, 'models/knn_best_model.pkl')
    joblib.dump(svr_tuned.model, 'models/svr_best_model.pkl')
    joblib.dump(preprocessor.scaler, 'models/scaler.pkl')
    joblib.dump(preprocessor.label_encoders, 'models/label_encoders.pkl')
    
    print("\nModels saved to 'models/' directory")
    
    return knn_tuned, svr_tuned


if __name__ == "__main__":
    train_with_tuning()
