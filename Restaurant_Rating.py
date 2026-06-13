import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

df = pd.read_csv("Dataset .csv")

# print(df.head())
# print(df.tail())

df.fillna(0, inplace=True)
df.fillna("Unknown")

# print(df.isnull().sum())

