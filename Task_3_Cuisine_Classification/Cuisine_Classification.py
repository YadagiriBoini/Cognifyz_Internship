import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import(
    accuracy_score,
    classification_report,
)

df = pd.read_csv("Dataset .csv")

# print(df.head())

df["Cuisines"] = df["Cuisines"].fillna("Unknown")

df["Cuisines"] = df["Cuisines"].apply(
    lambda x: x.split(",")[0].strip()
)

# print(df.isnull().sum())

df = df[
    [
        "City",
        "Average Cost for two",
        "Price range",
        "Votes",
        "Has Table booking",
        "Has Online delivery",
        "Cuisines"
    ]
]

le = LabelEncoder()

for col in df.select_dtypes(include="object"):
    df[col] = le.fit_transform(df[col].astype(str))

X = df.drop("Cuisines", axis=1)
y = df["Cuisines"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
) 

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

# print("Accuracy:", accuracy)

print(
    classification_report(
        y_test,
        y_pred
    )
)

print("Unique cuisines:", df["Cuisines"].nunique())
print(df["Cuisines"].value_counts().head(10))