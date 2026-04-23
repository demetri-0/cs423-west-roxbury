"""
West Roxbury Housing Analysis
Replicating Chapter 2 slides 15-43
"""

import pandas as pd
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
