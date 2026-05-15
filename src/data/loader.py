"""Data loading module"""

import pandas as pd
from pathlib import Path


class DataLoader:
    """Handle data loading operations"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        
    def load_data(self) -> pd.DataFrame:
        """Load diamond dataset"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        
        # Rename columns for clarity
        df.columns = ['id', 'carat', 'cut', 'color', 'clarity', 
                     'depth', 'table', 'price', 'x', 'y', 'z']
        
        return df
    
    def get_basic_info(self, df: pd.DataFrame):
        """Print basic dataset information"""
        print("Dataset Shape:", df.shape)
        print("\nMissing Values:\n", df.isna().sum())
        print("\nDuplicates:", df.duplicated().sum())
        print("\nData Types:\n", df.dtypes)