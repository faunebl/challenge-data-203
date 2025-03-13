# Challenge Data SNCF 

## Context 
"Transilien SNCF Voyageurs is the operator of suburban trains in Île-de-France. We operate over 6,200 trains, enabling 3.4 million passengers to travel. Each day, our trains serve thousands of stops. To ensure your journey goes as smoothly as possible, we provide an estimated waiting time, in minutes, as shown on the right in Figure 1. We aim to explore the possibility of improving the accuracy of waiting time forecasts."
Taken from the [Challenge Data website](https://challengedata.ens.fr/participants/challenges/166/).

## Usage 

### Importing the data 
Download the data from the link above after registering for the challenge, then put them in a folder called "data". 
To visualize a dataframe, use : 

```
from utils import PathsData, import_data
df = import_data(PathsData.X_TEST.value)
df.describe()
```