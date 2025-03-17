from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

class ModelEnum:
    LinearRegression = LinearRegression(),
    Ridge =  Ridge(),
    Lasso = Lasso(),
    ElasticNet = ElasticNet(),
    DecisionTree = DecisionTreeRegressor(),
    RandomForest = RandomForestRegressor(),
    GradientBoosting = GradientBoostingRegressor(),
    SVR = SVR(),
    KNN = KNeighborsRegressor(),
    MLP = MLPRegressor()

class Model:
    def __init__(self, model: ModelEnum):
        pass
