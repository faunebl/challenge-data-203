import logging
import time
from enum import Enum
# from tqdm import tqdm
# import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score

import optuna
import numpy as np
import dask
from dask.distributed import Client
from joblib import parallel_backend
from xgboost import XGBRegressor
import torch.nn as nn
import torch.optim as optim
import torch

client = Client()

logging.basicConfig(
    filename="model_training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger()

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
    XGBoost = XGBRegressor
    PyTorch = "PyTorch"

class ParamGridEnum(Enum):
    LinearModels = {
        "alpha": [0.01, 0.1, 1, 10, 100]
    }
    
    DecisionTree = {
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

    GradientBoosting = {
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
        "learning_rate": ["constant", "adaptive"],
        "max_iter": [50, 100, 200],
        "early_stopping": [True]
    }

    XGBoost = {
        "n_estimators": [100, 200, 500],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 10],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }

    PyTorch = {
        "hidden_layer_sizes": [50, 100, 200],
        "learning_rate": [0.0001, 0.001, 0.01],
        "epochs": [10, 20, 50],
        "batch_size": [16, 32, 64],
        "optimizer": ["adam", "sgd"]
    }

    @staticmethod
    def get_param_grid(model_enum):
        """Return the correct hyperparameter grid for a given model."""
        if model_enum in [ModelEnum.LinearRegression, ModelEnum.Ridge, ModelEnum.Lasso, ModelEnum.ElasticNet]:
            return ParamGridEnum.LinearModels.value
        elif model_enum == ModelEnum.DecisionTree:
            return ParamGridEnum.DecisionTree.value
        elif model_enum == ModelEnum.RandomForest:
            return ParamGridEnum.RandomForest.value
        elif model_enum == ModelEnum.GradientBoosting:
            return ParamGridEnum.GradientBoosting.value
        elif model_enum == ModelEnum.SVR:
            return ParamGridEnum.SVR.value
        elif model_enum == ModelEnum.KNN:
            return ParamGridEnum.KNN.value
        elif model_enum == ModelEnum.MLP:
            return ParamGridEnum.MLP.value
        elif model_enum == ModelEnum.XGBoost:
            return ParamGridEnum.XGBoost.value
        elif model_enum == ModelEnum.PyTorch:
            return ParamGridEnum.PyTorch.value
        else:
            raise ValueError(f"No parameter grid available for model: {model_enum}")

class Model:
    def __init__(self, model_enum, params=None):
        self.model_enum = model_enum
        self.params = params
        self.model = self.initialize_model()
        logger.info(f"Initialized model: {self.model_enum}")

    def initialize_model(self):
        if self.model_enum == ModelEnum.PyTorch:
            return self._initialize_pytorch_model()
        elif self.params:
            if self.model_enum == ModelEnum.XGBoost:
                return self.model_enum.__class__.XGBoost.value.set_params(self, **self.params)
            return self.model_enum.__class__(**self.params)
        return self.model_enum.value()

    def _initialize_pytorch_model(self):
        class SimpleNN(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, 50)
                self.fc2 = nn.Linear(50, 1)
                self.relu = nn.ReLU()

            def forward(self, x):
                return self.fc2(self.relu(self.fc1(x)))

        return SimpleNN

    def fit(self, X_train, y_train):
        try:
            start_time = time.time()
            if self.model_enum == ModelEnum.PyTorch:
                return self._fit_pytorch(X_train, y_train)

            self.model.fit(X_train, y_train)
            elapsed_time = time.time() - start_time
            logger.info(f"Model training completed in {elapsed_time:.2f} seconds.")
        except Exception as e:
            logger.error(f"Error during model training: {e}")

    def _fit_pytorch(self, X_train, y_train):
        input_dim = X_train.shape[1]
        model = self._initialize_pytorch_model()(input_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

        for epoch in range(100):
            optimizer.zero_grad()
            output = model(X_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()

        self.model = model

    def predict(self, X_test):
        if self.model_enum == ModelEnum.PyTorch:
            X_test_torch = torch.tensor(X_test, dtype=torch.float32)
            return self.model(X_test_torch).detach().numpy().flatten()
        else:
            return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        try:
            y_pred = self.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test,y_pred)
            logger.info(f"Model Evaluation - MSE: {mse:.4f}, R2: {r2:.4f}, MAE: {mae:.4f}")
            print(f"Model Evaluation - MSE: {mse:.4f}, R2: {r2:.4f}, MAE: {mae:.4f}")
            return mse, r2, mae
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")

    def grid_search(self, X_train, y_train, param_grid: ParamGridEnum = None, cv=3):
        if param_grid is None:
            param_grid = ParamGridEnum.get_param_grid(model_enum=self.model_enum)
        logger.warning(f"{'---------' * 10}")
        logger.warning("Starting GridSearchCV.")
        try:
            with parallel_backend("dask"):
                search = GridSearchCV(self.model, param_grid, cv=cv, verbose=3)
                search.fit(X_train, y_train)
                for i, params in enumerate(search.cv_results_["params"]):
                    mean_score = search.cv_results_["mean_test_score"][i]
                    logger.info(f"GridSearch Iteration {i+1}: {params} -> Score: {mean_score:.4f}")

            self.model = search.best_estimator_
            logger.info(f"Grid Search completed. Best parameters: {search.best_params_}")
            logger.warning(f"{'---' * 10}")
            return search.best_params_
        except Exception as e:
            logger.error(f"Error during Grid Search: {e}")
            logger.warning(f"{'---' * 10}")

    def optimize_hyperparams_optuna(self, X_train, y_train):
        logger.warning(f"{'---' * 10}")
        logger.info(f"Starting Optuna hyperparameter optimization for {self.model_enum}.")
        logger.warning(f"{'---' * 10}")

        def objective(trial):
            try:
                param_grid = ParamGridEnum.get_param_grid(self.model_enum)
                params = {key: getattr(trial, "suggest_categorical")(key, values) if isinstance(values, list) 
                        else getattr(trial, "suggest_loguniform")(key, *values) 
                        if isinstance(values, tuple) and len(values) == 2 else trial.suggest_int(key, *values) 
                        for key, values in param_grid.items()}
                
                if self.model_enum == ModelEnum.XGBoost:
                    model = self.model_enum.__class__.XGBoost.value.set_params(self, **self.params)
                else:
                    model = self.model_enum.__class__(**params)
                score = np.mean(cross_val_score(model, X_train, y_train, cv=3, scoring="neg_mean_squared_error"))

                logger.info(f"Optuna Trial {trial.number}: {params} -> Score: {score:.4f}")
                return score
            except Exception as e:
                logger.error(f"Error during Optuna trial {trial.number}: {e}")
                logger.warning(f"{'---' * 10}")
                return float("inf")

        try:
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=20)

            logger.info(f"Optuna optimization completed. Best parameters: {study.best_params}")
            logger.warning(f"{'---' * 10}")
            self.model = self.model_enum.__class__(**study.best_params)
            self.model.fit(X_train, y_train)
            return study.best_params
        except Exception as e:
            logger.error(f"Error during Optuna hyperparameter tuning: {e}")



