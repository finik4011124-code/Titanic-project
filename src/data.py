import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def compute_fill_stats(train_raw: pd.DataFrame) -> dict:
    return {
        "age_median": train_raw["Age"].median(),
        "fare_median": train_raw["Fare"].median(),
        "embarked_mode": train_raw["Embarked"].mode()[0],
    }


def preprocess(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    df = df.copy()
    df["Age"] = df["Age"].fillna(stats["age_median"])
    df["Fare"] = df["Fare"].fillna(stats["fare_median"])
    df["Embarked"] = df["Embarked"].fillna(stats["embarked_mode"])
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    return df


def align_columns(df: pd.DataFrame, reference_columns) -> pd.DataFrame:
    return df.reindex(columns=reference_columns, fill_value=0)