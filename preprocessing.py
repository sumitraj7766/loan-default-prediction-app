import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    # target convert
    if 'Default' in df.columns:
        df['Default'] = df['Default'].map({'Y':1, 'N':0})

    # drop id
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)

    #  HANDLE MISSING VALUES
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])  # categorical
        else:
            df[col] = df[col].fillna(df[col].median())   # numeric

    return df