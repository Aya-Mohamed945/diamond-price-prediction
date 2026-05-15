"""SVR Model Module"""

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import numpy as np


class SVRModel:
    """Support Vector Regression Model"""
    
    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1):
        self.model = SVR(kernel=kernel, C=C, epsilon=epsilon)
        
    def train(self, X_train, y_train):
        """Train the SVR model"""
        self.model.fit(X_train, y_train)
        return self.model
    
    def predict(self, X_test):
        """Make predictions"""
        return self.model.predict(X_test)
    
    def hyperparameter_tuning(self, X_train, y_train, param_grid=None):
        """Perform grid search for hyperparameter tuning"""
        if param_grid is None:
            param_grid = {
                'kernel': ['rbf', 'linear'],
                'C': [0.1, 1, 10, 100],
                'epsilon': [0.01, 0.1, 0.5, 1]
            }
        
        grid_search = GridSearchCV(
            SVR(), param_grid, 
            cv=5, scoring='r2', 
            n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_:.4f}")
        
        self.model = grid_search.best_estimator_
        return grid_search
