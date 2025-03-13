import polars as pl
import polars.selectors as cs
import itertools

class FeaturesFrame(pl.DataFrame):
    def __init__(
        self,
        data=None,
        schema=None,
        *,
        schema_overrides=None,
        strict=True,
        orient=None,
        infer_schema_length=...,
        nan_to_null=False,
    ):
        super().__init__(
            data,
            schema,
            schema_overrides=schema_overrides,
            strict=strict,
            orient=orient,
            infer_schema_length=infer_schema_length,
            nan_to_null=nan_to_null,
        )

    @classmethod
    def add_feature_interactions(cls):
        return cls.with_columns(
            [
                (pl.col(a) * pl.col(b)).alias(f"{a}_{b}_multiplied")
                for a, b in itertools.combinations(cls.select(cs.numeric()).columns, 2)
            ]
        )

    @classmethod
    def add_feature_log(cls):
        return cls.with_columns(
            [
                pl.when(pl.col(c) > 0)
                .then(pl.col(c).log())
                .otherwise(None)
                .alias(f"{c}_log")
                for c in cls.select(cs.numeric()).columns
            ]
        )

    @classmethod
    def add_feature_square(cls):
        return cls.with_columns(
            [pl.col(c).pow(2).alias(f"{c}_square") for c in cls.select(cs.numeric()).columns]
        )
    
    @classmethod
    def add_feature_sqrt(cls):
        return cls.with_columns(
            [pl.col(c).sqrt().alias(f"{c}_square") for c in cls.select(cs.numeric()).columns]
        )

    @classmethod
    def optimize_categorical(cls):
        return cls.with_columns(
            [
                pl.col(c).cast(pl.Categorical).alias(c)
                for c in cls.select(cs.string()).columns
            ]
        )

    @classmethod
    def encode_one_hot(cls):
        for col in cls.select(cs.string()).columns:
            cls = cls.with_columns(pl.select(col).to_dummies(prefix=col)).drop(col)
        return cls
