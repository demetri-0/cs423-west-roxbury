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
