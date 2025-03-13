import polars as pl
import polars.selectors as cs
import itertools

#change to oop -> 

def add_feature_interactions(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [
            (pl.col(a) * pl.col(b)).alias(f"{a}_{b}_interaction")
            for a, b in itertools.combinations(df.select(cs.numeric()).columns, 2)
        ]
    )

def add_feature_log(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [
            pl.when(pl.col(c) > 0)
            .then(pl.col(c).log())
            .otherwise(None)
            .alias(f"{c}_log")
            for c in df.select(cs.numeric()).columns
        ]
    )

def add_feature_square(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [
            pl.col(c).pow(2).alias(f"{c}_square") for c in df.select(cs.numeric()).columns
        ]
    )

def optimize_categorical(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [
            pl.col(c).cast(pl.Categorical).alias(c) for c in df.select(cs.string()).columns
        ]
    )

def encode_one_hot(df: pl.DataFrame) -> pl.DataFrame:
    for col in df.select(cs.string()).columns:
        df = df.with_columns(pl.select(col).to_dummies(prefix=col)).drop(col)
    return df