import polars as pl
import polars.selectors as cs
import itertools
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

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
            [pl.col(c).sqrt().alias(f"{c}_sqrt") for c in self.select(cs.numeric()).columns]
        ))


    def encode_one_hot(self):
        string_cols = self.select(cs.string()).columns
        if not string_cols:
            return type(self)(self)

        ohe = OneHotEncoder(handle_unknown='ignore')
        encoded_array = ohe.fit_transform(self.to_pandas()[string_cols])
        encoded_feature_names = ohe.get_feature_names_out(string_cols)
        encoded_df = pl.DataFrame(
            {
                name: encoded_array[:, i] for i, name in enumerate(encoded_feature_names)
            }
        )

        return type(self)(encoded_df)
