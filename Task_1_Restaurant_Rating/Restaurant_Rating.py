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

cols_to_drop = [
    "Restaurant ID",
    "Restaurant Name",
    "Address",
    "Locality",
    "Locality Verbose",
    "Rating color",
    "Rating text"
]

df = df.drop(cols_to_drop, axis=1)

# print(df.isnull().sum())

encoder = LabelEncoder()

for column in df.select_dtypes(include="object"):
    df[column] = encoder.fit_transform(df[column].astype(str))

y = df['Aggregate rating']
x = df.drop('Aggregate rating', axis=1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42 )

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE",mse)
print("R2 Score",r2)

importance = pd.DataFrame({
    'Feature': x.columns,
    'Coefficient': model.coef_
})

print(
    importance.sort_values(
        by= 'Coefficient',
        ascending=False
    )
)
