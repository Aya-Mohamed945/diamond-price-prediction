"""Data preprocessing module"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """Handle data preprocessing operations"""
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        categorical_cols = ['cut', 'color', 'clarity']
        df_encoded = df.copy()
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col])
            self.label_encoders[col] = le
            
        return df_encoded
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare features and target"""
        feature_columns = ['carat', 'depth', 'table', 'x', 'y', 'z',
                          'cut_encoded', 'color_encoded', 'clarity_encoded']
        
        X = df[feature_columns]
        y = df['price']
        
        return X, y, feature_columns
    
    def split_data(self, X, y):
        """Split data into train and test sets"""
        return train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
    
    def scale_features(self, X_train, X_test):
        """Scale features using StandardScaler"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
