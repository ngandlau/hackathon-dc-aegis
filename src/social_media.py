from pathlib import Path
from typing import Literal

import pandas as pd  # type: ignore


def fetch_disease_data(disease: Literal["flu", "measles"]) -> pd.DataFrame:
    df = None
    if disease == "flu":
        df = fetch_flu_data()
    elif disease == "measles":
        df = fetch_measles_data()
    else:
        raise ValueError(f"Invalid disease: {disease}")
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "social_media_value"]]


def fetch_flu_data() -> pd.DataFrame:
    FILE_PATH = Path("data", "flu_social_media_15k.csv")
    df = pd.read_csv(FILE_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_measles_data() -> pd.DataFrame:
    FILE_PATH = Path("data", "measles_social_media_volume.csv")
    df = pd.read_csv(FILE_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df
