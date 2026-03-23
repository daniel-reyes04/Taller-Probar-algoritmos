import pandas as pd

def load_and_prepare_data(path):
    df = pd.read_csv(path)
    df = df.dropna()

    X = df[[
        "Number_of_Customers_Per_Day",
        "Average_Order_Value",
        "Location_Foot_Traffic",
        "Marketing_Spend_Per_Day",
        "Number_of_Employees",
        "Operating_Hours_Per_Day"
    ]]

    y = df["Daily_Revenue"]

    return X, y
