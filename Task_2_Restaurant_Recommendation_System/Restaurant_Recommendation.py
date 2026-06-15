import pandas as pd

df = pd.read_csv("Dataset .csv")

df["Cuisines"] = df["Cuisines"].fillna("Unknown")

# print(df.isnull().sum())

df = df[[
    "Restaurant Name",
    "City",
    "Cuisines",
    "Price range",
    "Aggregate rating"
]]

def recommendation_restaurants(city, cuisines, price_range):
    results = df[
        (df["City"].str.contains(city, case=False, na=False))
        &
        (df['Cuisines'].str.contains(cuisines, case=False, na=False))
        &
        (df["Price range"] == price_range )
    ]

    results = results.sort_values(
        by= "Aggregate rating",
        ascending=False
    )

    if results.empty:
        return "No Restaurant Found"

    return results[
        [
            "Restaurant Name",
            "City",
            "Cuisines",
            "Price range",
            "Aggregate rating"
        ]
    ].head(10)

# print(df.head())

recommendations = recommendation_restaurants(
    # city="New Delhi",
    # cuisines="North Indian",
    # price_range=2

    # city="Mumbai",
    # cuisines = "Chinese",
    # price_range = 3

    city="Hyderabad",
    cuisines="Biryani",
    price_range=2
)

print(recommendations)
