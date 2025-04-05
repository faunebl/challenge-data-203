# Challenge Data SNCF 

## Context 
"Transilien SNCF Voyageurs is the operator of suburban trains in Île-de-France. We operate over 6,200 trains, enabling 3.4 million passengers to travel. Each day, our trains serve thousands of stops. To ensure your journey goes as smoothly as possible, we provide an estimated waiting time, in minutes, as shown on the right in Figure 1. We aim to explore the possibility of improving the accuracy of waiting time forecasts."
Taken from the [Challenge Data website](https://challengedata.ens.fr/participants/challenges/166/).

We can plot the correlation between the numerical features:

![alt text](https://github.com/faunebl/challenge-data-203/blob/master/feature_corr_mat.png?raw=true)

## Usage 

### Importing the data 
Download the data from the link above after registering for the challenge, then put them in a folder called "data". 
To visualize a dataframe, use : 

```python
from utils import PathsData, import_data
df = import_data(PathsData.X_TEST.value)
df.describe()
```

### Splitting data 
We can split the data into test and train using :

```python
from utils import split_data 
x = import_data(PathsData.X_TRAIN.value)
y = import_data(PathsData.Y_TRAIN.value)
x_train, x_test, y_train, y_test = split_data(x=x, y=y)
```

### Feature engineering
We can encode variables and get new variables with the features.py module. 
Example usage is in main.ipynb.

The way this module works is under the FeaturesFrame object : it inherits from a polars dataframe, adding functions to encode variables, multiply them together, get the square, the log etc. We can then stack these functions together in order to perform multiple transformations on our data. 

For example :

```python
from features import FeaturesFrame
FeaturesFrame(x_train).encode_label(encoder='label').add_feature_square().add_feature_interactions().scale_standard(set="train")
```
This single line encodes the labels, then squares all features, then multiplies them together and then scales the features. 

### Model selection
We can select the best models using the models.py module. Example usage is in the main.ipynb notebook. 

We have defined a Models class, so that we can try each model iteratively by only changing the name in the Enum. This makes it very easy to try a lot of models in a notebook. 

For example:
```python
from models import Model
model = Model(model_enum=ModelEnum.XGBoost)
model_fit = model.fit(test_train.to_numpy(), y_train=y_train.to_numpy())
model.evaluate(X_test=test_test.to_numpy(), y_test=y_test.to_numpy())
```
We can also run grid searches and hyperparameter optimization with optuna.

```python
model = Model(model_enum=ModelEnum.XGBoost)
params_opti_mlp = model.optimize_hyperparams_optuna(test_train.to_pandas(), y_train=y_train.to_pandas())
```
We can then change the "XGBoost" enum value by any of the models defined in the enum (see models.py)