import pandas as pd

def feature_engineering(df):
    df = pd.get_dummies(df, drop_first=True)
    return df