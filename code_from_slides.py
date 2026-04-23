""" Preliminary Exploration """

import pandas as pd
import mlba  #utilities for MLBA book

# Load data
housing_df = pd.read_csv('WestRoxbury.csv')

# Load data using MLBA utility
housing_df = mlba.load_data('WestRoxbury.csv')

housing_df.shape #find dimension of data frame
housing_df.head()  #show the 1st five rows
print(housing_df)  #show all the data


# Rename columns: replace spaces with '_' to allow dot notation
housing_df.columns = [s.strip().replace(' ', '_') for s in 
     housing_df.columns] # all columns

# Practice showing the first four rows of the data
housing_df.loc[0:3] # loc[a:b] gives rows a to b, inclusive and allows labels
housing_df.iloc[0:4] # iloc[a:b] gives rows a to b-1 and requires integers

""" Sampling and Oversampling """

# random sample of 5 houses
housing_df.sample(5)

# oversample houses with over 10 rooms
weights = [0.9 if rooms > 10 else 0.01 for rooms in 
     housing_df.ROOMS]
housing_df.sample(5, weights=weights)

# stratified sampling of houses by remodeling state
housing_df.groupby('REMODEL', dropna=False).sample(frac=0.8)

""" Reviewing Variables """

# random sample of 5 houses
housing_df.sample(5)

# oversample houses with over 10 rooms
weights = [0.9 if rooms > 10 else 0.01 for rooms in 
     housing_df.ROOMS]
housing_df.sample(5, weights=weights)

# stratified sampling of houses by remodeling state
housing_df.groupby('REMODEL', dropna=False).sample(frac=0.8)

""" Data Prep """

housing_df.REMODEL = housing_df.REMODEL.fillna('None').astype('category')
housing_df.REMODEL.cat.categories 
housing_df.REMODEL.dtype 

""" Create Binary Dummies """

# the missing values will create a third category
# use the arguments drop_first and dummy_na to control the outcome
housing_df = pd.get_dummies(housing_df, prefix_sep='_', dummy_na=True, 
     dtype=int)
housing_df.loc[:, 'REMODEL_None':'REMODEL_Recent'].head(5)

""" Replacing Missing Data With Median """

# To illustrate missing data procedures, we first convert a 
# few entries for BEDROOMS to NA's. Then we impute these 
# missing values using the median of the remaining values.

import numpy as np

missingRows = housing_df.sample(10).index
housing_df.loc[missingRows, 'BEDROOMS'] = np.nan
print('Number of rows with valid BEDROOMS values after setting to NA: ',
housing_df['BEDROOMS'].count())

# remove rows with missing values
reduced_df = housing_df.dropna()
print('Number of rows after removing rows with missing values: ', len(reduced_df))

# replace the missing values using the median of the remaining values.
medianBedrooms = housing_df['BEDROOMS'].median()
housing_df.BEDROOMS = housing_df.BEDROOMS.fillna(value=medianBedrooms)
print('Number of rows with valid BEDROOMS values after filling NA values: ',
   housing_df['BEDROOMS'].count())

""" Normalizing and Rescaling """

# Normalizing a data frame, pandas:
# pandas:
norm_df = (housing_df - housing_df.mean()) / housing_df.std()
# Normalizing a data frame, scikit-learn:
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
norm_df = pd.DataFrame(scaler.fit_transform(housing_df),
   index=housing_df.index, columns=housing_df.columns)
# result of the transformation is a numpy array, we convert it into a data frame


# Rescaling a data frame, pandas:
norm_df = (housing_df - housing_df.min()) / (housing_df.max() - housing_df.min())
# Rescaling a data frame, scikit-learn:
scaler = MinMaxScaler()
norm_df = pd.DataFrame(scaler.fit_transform(housing_df),
   index=housing_df.index, columns=housing_df.columns)

""" Partitioning Data """

from sklearn.model_selection import train_test_split
trainData, holdoutData = train_test_split(housing_df, 
   test_size=0.40, random_state=1)

""" Clean and Preprocess """

# data loading and preprocessing
housing_df = mlba.load_data('WestRoxbury.csv')
housing_df.columns = [s.strip().replace(' ', '_') for s in housing_df.columns]
housing_df.REMODEL = housing_df.REMODEL.fillna('None').astype('category')
housing_df = pd.get_dummies(housing_df, prefix_sep='_', drop_first=True)

# create list of predictors and outcome
excludeColumns = ('TOTAL_VALUE', 'TAX')
predictors = [column for column in housing_df.columns if column not in 
   excludeColumns]
outcome = 'TOTAL_VALUE'

""" Partition """

# partition data
X = housing_df[predictors]
y = housing_df[outcome]
train_X, holdout_X, train_y, holdout_y = train_test_split(X, y, 
    test_size=0.4, random_state=1)

""" Fit Model and Score """

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(train_X, train_y)

holdout_pred = model.predict(holdout_X)
holdout_results = pd.DataFrame({
'TOTAL_VALUE': holdout_y,
'predicted': holdout_pred,
'residual': holdout_y - holdout_pred,
})
holdout_results.head()

""" Assess Accuracy """

# import the utility function regressionSummary
from mlba import regressionSummary

# holdout set
regressionSummary(y_true=holdout_results.TOTAL_VALUE, 
    y_pred=holdout_results.predicted)
