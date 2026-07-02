import pandas as pd

TITLE_MAP = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
}


def add_title(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Title"] = df["Name"].str.extract(r",\s*([^.]*)\.")
    df["Title"] = df["Title"].map(TITLE_MAP).fillna("Rare")
    return df


def add_family_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df


def add_deck(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Deck"] = df["Cabin"].str[0].fillna("Unknown")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_title(df)
    df = add_family_features(df)
    df = add_deck(df)
    df = pd.get_dummies(df, columns=["Embarked", "Title", "Deck"], dtype=int)
    return df


def select_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in ["PassengerId", "Survived", "Name", "Ticket", "Cabin"] if c in df.columns]
    return df.drop(columns=drop_cols)