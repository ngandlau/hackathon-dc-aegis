from typing import Literal

import pandas as pd  # type: ignore
from delphi_epidata import Epidata  # type: ignore

from src.utils import _convert_epiweek_to_date


def fetch_flu_data() -> pd.DataFrame:
    regions = ["nat"]
    epiweeks = Epidata.range(202501, 202524)  # Year 2023 weeks 1 to 24
    response = Epidata.fluview(regions, epiweeks)
    if response["result"] != 1:
        raise Exception(f"API request failed: {response['message']}")
    df = pd.DataFrame(response["epidata"])
    return df


def mock_fetch_flu_data() -> pd.DataFrame:
    df = pd.read_csv("data/flu_clean.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


def _clean_flu_data(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    df_clean["date"] = df_clean["epiweek"].apply(_convert_epiweek_to_date)
    df_clean["cases"] = df_clean["num_ili"]
    df_clean = df_clean[["date", "cases"]].copy()
    df_clean = df_clean.sort_values("date")
    df_clean.reset_index(drop=True, inplace=True)
    return df_clean[["date", "cases"]]


def fetch_disease_data(
    disease: Literal["flu", "measles"], mock: bool = True
) -> pd.DataFrame:
    df = None

    if disease == "flu":
        if mock:
            df = mock_fetch_flu_data()
        else:
            df = fetch_flu_data()
            df = _clean_flu_data(df)  # cols: date, cases
    elif disease == "measles":
        df = pd.read_csv("data/measles.csv")  # cols: date, cases
    else:
        raise ValueError(f"Invalid disease: {disease}")

    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "num_cases"]]
