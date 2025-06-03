from pathlib import Path
from typing import Literal

import pandas as pd


def fetch_disease_data(disease: Literal["flu", "measles"]) -> pd.DataFrame:
    if disease == "flu":
        return fetch_flu_data()
    elif disease == "measles":
        return fetch_measles_data()
    else:
        raise ValueError(f"Invalid disease: {disease}")


def fetch_flu_data() -> pd.DataFrame:
    FILE_PATH = Path("data", "flu_social_media_15k.csv")
    df = pd.read_csv(FILE_PATH)
    return df


def fetch_measles_data() -> pd.DataFrame:
    FILE_PATH = Path("data", "measles_social_media_volume.csv")
    df = pd.read_csv(FILE_PATH)
    return df
