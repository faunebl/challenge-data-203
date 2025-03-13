from enum import Enum
import polars as pl
import polars.selectors as cs
import matplotlib.pyplot as plt 
import seaborn as sns

class PathsData(Enum):
    X_TRAIN = r"data\x_train_final.csv"
    X_TEST = r"data\x_test_final.csv"
    Y_TRAIN = r"data\y_train_final_j5KGWWK.csv"

def import_data(path: PathsData) -> pl.DataFrame:
    return pl.read_csv(path).drop(cs.contains("Unnamed"), "")

def split_data(x: pl.DataFrame, y: pl.DataFrame, test_size=0.2):
    split_index = int(x.height * (1 - test_size))
    x_train = x[:split_index]
    x_test = x[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]
    return x_train, x_test, y_train, y_test

def plot_correlation(df: pl.DataFrame):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        df.select(cs.numeric()).corr(), 
        annot=True, 
        fmt='.2f', 
        linewidths=0.5, 
        center=0, 
        cmap=sns.cubehelix_palette(
            start=.5, 
            rot=-.5, 
            dark=0.1, 
            light=1.5, 
            hue=1,
            as_cmap=True
            )
    )
    plt.title("Feature Correlation Matrix")
    plt.show()