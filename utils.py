from enum import Enum
import polars as pl
import polars.selectors as cs

class PathsData(Enum):
    X_TRAIN = r"data\x_train_final.csv"
    X_TEST = r"data\x_test_final.csv"
    Y_TRAIN = r"Cdata\y_train_final_j5KGWWK.csv"

def import_data(path: PathsData) -> pl.DataFrame:
    return pl.read_csv(path).drop(cs.contains("Unnamed"), "")