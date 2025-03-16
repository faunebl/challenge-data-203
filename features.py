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

    def add_feature_interactions(self):
        return type(self)(self.with_columns(
            [
                (pl.col(a) * pl.col(b)).alias(f"{a}_{b}_multiplied")
                for a, b in itertools.combinations(self.select(cs.numeric()).columns, 2)
            ]
        ))

    def add_feature_log(self):
        return type(self)(self.with_columns(
            [
                pl.when(pl.col(c) > 0)
                .then(pl.col(c).log())
                .otherwise(None)
                .alias(f"{c}_log")
                for c in self.select(cs.numeric()).columns
            ]
        ))

    def add_feature_square(self):
        return type(self)(self.with_columns(
            [pl.col(c).pow(2).alias(f"{c}_square") for c in self.select(cs.numeric()).columns]
        ))
    
    def add_feature_sqrt(self):
        return type(self)(self.with_columns(
            [pl.col(c).sqrt().alias(f"{c}_square") for c in self.select(cs.numeric()).columns]
        ))

    # @classmethod
    # def optimize_categorical(self):
    #     return self.with_columns(
    #         [
    #             pl.col(c).cast(pl.Categorical).alias(c)
    #             for c in self.select(cs.string()).columns
    #         ]
    #     )

    def encode_one_hot(self):
        for col in self.select(cs.string()).columns:
            self = self.with_columns(pl.select(col).to_dummies(prefix=col)).drop(col)
        return type(self)(self)
