import pandas as pd
from pathlib import Path


class DataLoader:

    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path, encoding="latin-1")
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        # drop unnamed columns
        df = df.drop(columns=[col for col in df.columns if "Unnamed" in col])

        # rename columns
        df = df.rename(columns={"v1": "label", "v2": "message"})

        # drop duplicates
        df = df.drop_duplicates()

        return df

    def summary(self, df: pd.DataFrame) -> None:
        print("shape:", df.shape)
        print("columns:", df.columns.tolist())
        print("label distribution:\n", df["label"].value_counts())
        print("null values:\n", df.isnull().sum())