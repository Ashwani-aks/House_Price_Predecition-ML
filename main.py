# import required libraries
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# load dataset
df = pd.read_csv("dataset_2.csv")
print(df.head())

# check and remove missing values
print(df.isnull().sum())
df.dropna(inplace=True)

# Data Refineing

# lowercase
text_columns = [
    "Location",
    "Street_Type",
    "Furnishing",
    "Property_Type",
    "Has_Pool"
]

df[text_columns] = df[text_columns].apply(lambda x: x.str.lower())

# splitting of data 
X = df[
    [
        "Location",
        "Street_Type",
        "Furnishing",
        "Property_Type",
        "Has_Pool",
        "Area_SqFt",
        "Rooms",
        "Build_Year"
    ]
]
y = df["Price"]

# columns in dataset
numerical_col = [
    "Area_SqFt",
    "Rooms",
    "Build_Year"
]

categorical_col = [
    "Location",
    "Street_Type",
    "Furnishing",
    "Property_Type",
    "Has_Pool"
]

# one hot encoder using ColumnTransformer
prep = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_col
        )
    ],
    remainder="passthrough"
)

# pipelineing
model = Pipeline(
    steps=[
        ("prep", prep),
        ("reg", LinearRegression())
    ]
)

# train split and test
x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# train and test
model.fit(x_train, y_train)

# model efficient working
prediction = model.predict(x_test)
print("MAE =", mean_absolute_error(y_test, prediction))
print("RMSE =", mean_squared_error(y_test, prediction) ** 0.5)
print("R2 Score =", r2_score(y_test, prediction))

# user input
area = float(input("Enter Area in Square Feet : "))
rooms = int(input("Enter Number of Rooms : "))
year = int(input("Enter Build Year : "))
location = input("Enter Location (city): ").lower()
street = input("Enter Street Type (Residential Lane, Corner Plot, Highway Facing, Main Road, Gated Society): ").lower()
furnishing = input("Enter Furnishing (Furnished, Semi-Furnished, Unfurnished) : ").lower()
property_type = input("Enter Property Type (Apartment, Duplex, Villa, Independent House): ").lower()
pool = input("Has Pool (yes/no) : ").lower()

new_house = pd.DataFrame({
    "Area_SqFt":[area],
    "Rooms":[rooms],
    "Build_Year":[year],
    "Location":[location],
    "Street_Type":[street],
    "Furnishing":[furnishing],
    "Property_Type":[property_type],
    "Has_Pool":[pool]
})

# prediction
predicted_price = model.predict(new_house)
print()
print("Predicted House Price = ", predicted_price[0])

# save model
joblib.dump(model, "HousePricePredictionModel.joblib")
print()
print("Model Saved Successfully")
