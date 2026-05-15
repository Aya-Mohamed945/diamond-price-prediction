"""Feature engineering module"""

import pandas as pd
import numpy as np


class FeatureEngineer:
    """Handle feature engineering operations"""
    
    @staticmethod
    def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features"""
        df_features = df.copy()
        
        # Carat to volume ratio
        df_features['carat_per_volume'] = df_features['carat'] / (
            df_features['x'] * df_features['y'] * df_features['z']
        )
        
        # Depth anomalies
        df_features['depth_anomaly'] = np.abs(
            df_features['depth'] - df_features['z'] / df_features['x'] * 100
        )
        
        return df_features
    
    @staticmethod
    def add_price_per_carat(df: pd.DataFrame) -> pd.DataFrame:
        """Add price per carat feature"""
        df_features = df.copy()
        df_features['price_per_carat'] = df_features['price'] / df_features['carat']
        return df_features
