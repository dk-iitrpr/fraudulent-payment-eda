import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    df_001 = pd.read_csv('data/dataset-part-001.csv')
    df_002 = pd.read_csv('data/dataset-part-002.csv')
    df_003 = pd.read_csv('data/dataset-part-003.csv')
    df_004 = pd.read_csv('data/dataset-part-004.csv')
    df_005 = pd.read_csv('data/dataset-part-005.csv')
    df = pd.concat([df_001,df_002,df_003,df_004,df_005], ignore_index=True)
    print(f"The shape of the dataset is {df.shape}")
    print(f"The column names of the dataset:\n{df.columns}")
    print(f"The head of the csv:\n{df.head}")
    
    feature_columns = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    target_column = ["isFraud"]
    print()
    print(f"The needed csv details:")
    print("Features:", feature_columns)
    print("Target:", target_column)
    print(df[feature_columns + target_column].head())

    model_df = df[feature_columns + target_column].copy()
    X = model_df[feature_columns]
    y = model_df[target_column]

    return X, y, list(X.columns)

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)