import pandas as pd
from ucimlrepo import fetch_ucirepo

def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df
def get_columns_names(df):
    return df.columns.tolist()
def load_sample_data():
    dataset = fetch_ucirepo(id = 2)
    df = dataset.data.features
    df["income"] = dataset.data.targets
    return df