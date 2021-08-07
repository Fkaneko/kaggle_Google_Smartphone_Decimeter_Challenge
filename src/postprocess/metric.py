from typing import Tuple, Union

import numpy as np
import pandas as pd


def calc_haversine(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
    return_azimuth: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Calculates the great circle distance between two points
    on the earth. Inputs are array-like and specified in decimal degrees.
    modify from
    https://www.kaggle.com/columbia2131/device-eda-interpolate-by-removing-device-en-ja
    """
    RADIUS = 6_367_000

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    dist = 2 * RADIUS * np.arcsin(a ** 0.5)

    if return_azimuth:
        # azimuth_np = np.arctan2(
        #     np.cos(lat2) * np.sin(dlon),
        #     np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon),
        # )
        # https://en.wikipedia.org/wiki/Great-circle_navigation

        azimuth = np.arctan2(
            np.cos(lat2) * np.sin(dlon),
            np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon),
        )
        azimuth = np.where(azimuth < 0, azimuth + np.pi * 2, azimuth)
        return dist, azimuth

    return dist


def calc_dist_from_dfs(df: pd.DataFrame, df_pred: pd.DataFrame) -> np.ndarray:
    dist = calc_haversine(
        lat1=df["latDeg"].values,
        lon1=df["lngDeg"].values,
        lat2=df_pred["latDeg"].values,
        lon2=df_pred["lngDeg"].values,
    )
    return dist


def add_date_feat(df: pd.DataFrame, target: str = "collectionName") -> pd.DataFrame:
    df["date"] = df[target].apply(lambda x: "-".join(x.split("-")[:3]))
    df["date"] = pd.to_datetime(df["date"])
    return df


def calc_2021_ratio(df: pd.DataFrame) -> float:
    len_2021 = (
        len(df[["date", "phone"]]).set_index("date")["2020-12-31":].value_counts()
    )
    len_all = len((df[["date", "phone"]]).value_counts())
    return len_2021 / len_all


def calc_metric(pred_df: pd.DataFrame, target_col: str = "dist_err") -> pd.DataFrame:
    val_df2 = pd.DataFrame()
    val_df2["phone"] = pred_df.phone.unique().tolist()
    val_df2["dist_50"] = [
        np.percentile(pred_df[pred_df.phone == ph][target_col], 50)
        for ph in val_df2["phone"].tolist()
    ]
    val_df2["dist_95"] = [
        np.percentile(pred_df[pred_df.phone == ph][target_col], 95)
        for ph in val_df2["phone"].tolist()
    ]
    val_df2["avg_dist_50_95"] = (val_df2["dist_50"] + val_df2["dist_95"]) / 2.0
    # print("Val evaluation details:\n", val_df2)
    print("Val evaluation details:\n", val_df2["avg_dist_50_95"].mean())
    return val_df2


def print_metric(df: pd.DataFrame):
    df["dist_err_orig"] = calc_haversine(
        df["latDeg_gt"], df["lngDeg_gt"], df["latDeg"], df["lngDeg"],
    )
    met_df = calc_metric(pred_df=df, target_col="dist_err_orig")
    return met_df
