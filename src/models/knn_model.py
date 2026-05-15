"""KNN Model Module"""

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV


class KNNModel:
    """K-Nearest Neighbors Regressor Model"""
    
    def __init__(self, n_neighbors=5, metric='euclidean'):
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors, metric=metric)
        
    def train(self, X_train, y_train):
        """Train the KNN model"""
        self.model.fit(X_train, y_train)
        return self.model
    
    def predict(self, X_test):
        """Make predictions"""
        return self.model.predict(X_test)
    
    def hyperparameter_tuning(self, X_train, y_train, param_grid=None):
        """Perform grid search for hyperparameter tuning"""
        if param_grid is None:
            param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }
        
        grid_search = GridSearchCV(
            KNeighborsRegressor(), param_grid,
            cv=5, scoring='r2',
            n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_:.4f}")
        
        self.model = grid_search.best_estimator_
        return grid_search
