"""Linear Regression Model Module"""

from sklearn.linear_model import LinearRegression
import pandas as pd


class LinearRegressionModel:
    """Multiple Linear Regression Model"""
    
    def __init__(self):
        self.model = LinearRegression()
        
    def train(self, X_train, y_train):
        """Train the linear regression model"""
        self.model.fit(X_train, y_train)
        return self.model
    
    def predict(self, X_test):
        """Make predictions"""
        return self.model.predict(X_test)
    
    def get_feature_importance(self, feature_names):
        """Get feature coefficients"""
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': self.model.coef_
        }).sort_values('Coefficient', key=abs, ascending=False)
        
        return importance_df
