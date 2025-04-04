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
    #remove outliers:
    x = x.with_columns(y).filter(pl.col('p0q0').ge(pl.col('p0q0').quantile(0.05)) & pl.col('p0q0').le(pl.col('p0q0').quantile(0.95)))
    y = x.select('p0q0')
    x = x.drop('p0q0')
    # split
    split_index = int(x.height * (1 - test_size))
    x_train = x[:split_index]
    x_test = x[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]
    return x_train, x_test, y_train, y_test

def plot_correlation(df: pl.DataFrame):
    plt.figure(figsize=(16, 15))
    sns.heatmap(
        df.select(cs.numeric()).corr(), 
        annot=True, 
        fmt='.2f', 
        linewidths=0.5, 
        center=0, 
        cmap=sns.color_palette('RdBu', n_colors=15, as_cmap=True),
        robust=True
    )
    plt.title("Feature Correlation Matrix")
    plt.show()