from enum import Enum
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

class ModelEnum(Enum):
    LinearRegression = LinearRegression
    Ridge =  Ridge
    Lasso = Lasso
    ElasticNet = ElasticNet
    DecisionTree = DecisionTreeRegressor
    RandomForest = RandomForestRegressor
    GradientBoosting = GradientBoostingRegressor
    SVR = SVR
    KNN = KNeighborsRegressor
    MLP = MLPRegressor

class ParamGridEnum(Enum):
    LinearModels = {
        "alpha": [0.01, 0.1, 1, 10, 100]
    }
    DecisionTree =  {
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    RandomForest = {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    GradientBoosting =  {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 10],
        "subsample": [0.8, 1.0]
    }

    SVR = {
        "C": [0.1, 1, 10, 100],
        "epsilon": [0.01, 0.1, 0.5],
        "kernel": ["linear", "rbf"]
    }

    KNN = {
        "n_neighbors": [3, 5, 10, 20],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"]
    }

    MLP = {
    "hidden_layer_sizes": [(50,), (100,), (100, 50)],
    "activation": ["relu", "tanh"],
    "alpha": [0.0001, 0.001, 0.01],
    "learning_rate": ["constant", "adaptive"]
}


class Model:
    def __init__(self, model_enum: ModelEnum, params: dict = None):
        self.model_enum = model_enum
        self.params = params
        self.model = self.initialize_model()

    def initialize_model(self):
        if self.params:
            return self.model_enum.value(**self.params)
        return self.model_enum.value()

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"R^2 Score: {r2:.4f}")
        return {"mse": mse, "mae": mae, "r2": r2}

    def visualize(self, X_test, y_test):
        predictions = self.predict(X_test)
        sns.scatterplot(x=y_test, y=predictions)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"{self.model_enum.name} Performance")
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
        plt.show()

    def optimize_parameters(self, X_train, y_train, param_grid: ParamGridEnum, cv: int = 5, scoring: str = 'neg_mean_squared_error'):
        grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid, cv=cv, scoring=scoring)
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        print(f"Best Parameters: {grid_search.best_params_}")
        return grid_search.best_params_

