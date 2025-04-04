from typing import Literal
import polars as pl
import polars.selectors as cs
import itertools
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

class EnhancedLabelEncoder(LabelEncoder): #adding support for unseen labels #! to use later
    def transform(self, y):
        y = np.array(y)
        mapping = {label: index for index, label in enumerate(self.classes_)}
        return np.array([mapping.get(item, -1) for item in y]) #we choose -1 as a value for all unseen labels -> use None ?

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

    def encode_label(self, encoder: Literal['label', 'frequency']):
        """!!! Only works with the initial x_train data (with date as str), do not use twice in a row

        Returns:
            FeaturesFrame: dataframe 
        """        
        self = type(self)(
            self
            .with_columns(pl.col('date').str.to_date("%Y-%m-%d"))
            .with_columns(
                pl.col('date').dt.weekday().alias('weekday'),
                # pl.col('date').dt.year().alias('year'), #! useless: only 2023
                pl.col('date').dt.month().alias('month'),
                pl.col('date').dt.day().alias('day')
            )
        )
        dict_string = self.select(cs.string()).to_dict(as_series=False)
        if encoder == 'label':
            return type(self)(   
                self
                .with_columns(
                    encoded_train = LabelEncoder().fit_transform(dict_string['train']),
                    encoded_gare = LabelEncoder().fit_transform(dict_string['gare'])
                )
                .drop(cs.string(), cs.date())
            )
        elif encoder == 'frequency':
            freqs_train = (
                self.group_by('train')
                .len(name='count')
                .select((pl.col('count') / self.height).alias('encoded_train'), 'train')
            )
            self = type(self)(self.join(freqs_train, on='train', how="left"))
            freqs_gare = (
                self.group_by('gare')
                .len(name='count')
                .select((pl.col('count') / self.height).alias('encoded_gare'), 'gare')
            )
            self = type(self)(self.join(freqs_gare, on='gare', how="left"))
            return self
        
    def remove_outliers(self, method: Literal['quantile'] = 'quantile', lower_quantile: float = 0.05, upper_quantile: float = 0.95):
        df_clean = self.clone()
        # columns = cs.numeric().
        if method=='quantile':
            lower_bounds = df_clean.select(cs.numeric().quantile(lower_quantile))
            upper_bounds = df_clean.select(cs.numeric().quantile(upper_quantile))
            
            print(f"lower bounds = {lower_bounds}, upper bounds = {upper_bounds}")
            
            df_clean = df_clean.filter((cs.numeric().ge(lower_bounds)) & (cs.numeric().le(upper_bounds)))
        
        return type(self)(df_clean)

    
    def scale_standard(self, set: Literal['train', 'test'] = 'train', train_scaler: StandardScaler = None):
        if set == 'test' and train_scaler is None:
            raise Exception(
                """
                Please compute the standard scaler first by using StandardScaler().fit(df.to_numpy()) where df is the train set. \n
                You can also run this function on the train set, it will return a tuple with (scaler, df).
                """
            )
        
        if set == 'train':
            scaler = StandardScaler().fit(self.to_numpy())
            return (scaler, type(self)(
                pl.DataFrame(
                    data = dict(zip(self.columns, scaler.transform(self.to_numpy()).transpose() )) 
                )
            ))
        else:
            return type(self)(
                pl.DataFrame(
                    data = dict(zip(self.columns, train_scaler.transform(self.to_numpy()).transpose() )) #! scaler fit que sur train 
                )
            )
