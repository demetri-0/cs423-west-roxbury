"""
West Roxbury Housing Analysis
Replicating Chapter 2 slides 15-43
"""

import pandas as pd
import numpy as np
import mlba


# Section 1: Load and inspect the data
# Defensive parsing checks that the file loaded correctly before analysis.
housing_df = (
    mlba.load_data("WestRoxbury.csv")
    .rename(columns=lambda col: col.strip().replace(" ", "_"))
)

print("Data shape:")
print(housing_df.shape)

print("\nColumn data types:")
print(housing_df.dtypes)

print("\nFirst five rows:")
print(housing_df.head())

print("\nMemory usage in MB:")
print(housing_df.memory_usage(deep=True).sum() / 1e6)

print("\nCleaned column names:")
print(list(housing_df.columns))


# Section 2: Row selection and sampling
# loc selects rows by label, while iloc selects rows by integer position.
print("\nFirst four rows using loc:")
print(housing_df.loc[0:3])

print("\nFirst four rows using iloc:")
print(housing_df.iloc[0:4])

print("\nRandom sample of five houses:")
print(housing_df.sample(5, random_state=1))

weighted_sample = housing_df.sample(
    5,
    weights=housing_df["ROOMS"].apply(lambda rooms: 0.9 if rooms > 10 else 0.01),
    random_state=1,
)

print("\nWeighted sample favoring houses with more than 10 rooms:")
print(weighted_sample)

stratified_sample = (
    housing_df
    .groupby("REMODEL", dropna=False, group_keys=False)
    .sample(frac=0.8, random_state=1)
)

print("\nStratified sample by REMODEL:")
print(stratified_sample.head())
print("Stratified sample shape:", stratified_sample.shape)


# Section 3: Reviewing variables
print("\nSelected variable data types:")
print(housing_df.dtypes[["TOTAL_VALUE", "FLOORS", "REMODEL"]])

print("\nVariable list:")
print(housing_df.columns)


# Section 4: Prepare categorical variables
# Missing REMODEL values are treated as their own category.
housing_with_remodel_category = (
    housing_df
    .assign(REMODEL=lambda df: df["REMODEL"].fillna("None").astype("category"))
)

print("\nREMODEL categories:")
print(housing_with_remodel_category["REMODEL"].cat.categories)

print("\nREMODEL data type:")
print(housing_with_remodel_category["REMODEL"].dtype)

housing_dummies_df = pd.get_dummies(
    housing_with_remodel_category,
    prefix_sep="_",
    dummy_na=True,
    dtype=int,
)

print("\nREMODEL dummy variables:")
print(housing_dummies_df.loc[:, "REMODEL_None":"REMODEL_Recent"].head())


# Section 5: Replacing missing data with the median
# This demonstrates dropping missing rows versus imputing with a typical value.
missing_bedrooms_df = housing_dummies_df.copy()

missing_rows = missing_bedrooms_df.sample(10, random_state=1).index
missing_bedrooms_df.loc[missing_rows, "BEDROOMS"] = np.nan

print("\nValid BEDROOMS values after setting 10 rows to missing:")
print(missing_bedrooms_df["BEDROOMS"].count())

reduced_df = missing_bedrooms_df.dropna()

print("\nRows after removing rows with missing values:")
print(len(reduced_df))

median_bedrooms = missing_bedrooms_df["BEDROOMS"].median()

missing_bedrooms_df = missing_bedrooms_df.assign(
    BEDROOMS=lambda df: df["BEDROOMS"].fillna(median_bedrooms)
)

print("\nValid BEDROOMS values after filling missing values with median:")
print(missing_bedrooms_df["BEDROOMS"].count())
