import copy
import warnings
from typing import List

import numpy as np
import pandas as pd
import scipy
import simdkalman
from numpy.fft import irfft, rfft, rfftfreq
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from src.postprocess.metric import calc_haversine

warnings.filterwarnings("ignore")


def apply_kf_smoothing(df: pd.DataFrame) -> pd.DataFrame:
    """
    from https://www.kaggle.com/emaerthin/demonstration-of-the-kalman-filter
    """

    def _get_kalman_filter() -> simdkalman.KalmanFilter:
        T = 1.0
        state_transition = np.array(
            [
                [1, 0, T, 0, 0.5 * T ** 2, 0],
                [0, 1, 0, T, 0, 0.5 * T ** 2],
                [0, 0, 1, 0, T, 0],
                [0, 0, 0, 1, 0, T],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        process_noise = (
            np.diag([1e-5, 1e-5, 5e-6, 5e-6, 1e-6, 1e-6]) + np.ones((6, 6)) * 1e-9
        )
        observation_model = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
        observation_noise = np.diag([5e-5, 5e-5]) + np.ones((2, 2)) * 1e-9

        kf = simdkalman.KalmanFilter(
            state_transition=state_transition,
            process_noise=process_noise,
            observation_model=observation_model,
            observation_noise=observation_noise,
        )
        return kf

    kf_ = _get_kalman_filter()
    unique_paths = df[["collectionName", "phoneName"]].drop_duplicates().to_numpy()
    for collection, phone in tqdm(unique_paths):
        cond = np.logical_and(
            df["collectionName"] == collection, df["phoneName"] == phone
        )
        data = df[cond][["latDeg", "lngDeg"]].to_numpy()
        data = data.reshape(1, len(data), 2)
        smoothed = kf_.smooth(data)
        df.loc[cond, "latDeg"] = smoothed.states.mean[0, :, 0]
        df.loc[cond, "lngDeg"] = smoothed.states.mean[0, :, 1]
    return df


def filter_outlier(df: pd.DataFrame, one_direction: bool = False) -> pd.DataFrame:
    """
    https://www.kaggle.com/dehokanta/baseline-post-processing-by-outlier-correction
    """
    df["dist_pre"] = 0
    df["dist_pro"] = 0

    df["latDeg_pre"] = df["latDeg"].shift(periods=1, fill_value=0)
    df["lngDeg_pre"] = df["lngDeg"].shift(periods=1, fill_value=0)
    df["latDeg_pro"] = df["latDeg"].shift(periods=-1, fill_value=0)
    df["lngDeg_pro"] = df["lngDeg"].shift(periods=-1, fill_value=0)
    df["dist_pre"] = calc_haversine(df.latDeg_pre, df.lngDeg_pre, df.latDeg, df.lngDeg)
    df["dist_pro"] = calc_haversine(df.latDeg, df.lngDeg, df.latDeg_pro, df.lngDeg_pro)

    # start, end fix
    list_phone = df["phone"].unique()
    for phone in list_phone:
        ind_s = df[df["phone"] == phone].index[0]
        ind_e = df[df["phone"] == phone].index[-1]
        df.loc[ind_s, "dist_pre"] = 0
        df.loc[ind_e, "dist_pro"] = 0

    # 95% tile
    pro_95 = df["dist_pro"].mean() + (df["dist_pro"].std() * 2)
    pre_95 = df["dist_pre"].mean() + (df["dist_pre"].std() * 2)
    # find outlier data
    if one_direction:
        targets = ["latDeg", "lngDeg"]
        dfs = []
        for phone, df_ in df.groupby("phone"):
            pre_mask = df_["dist_pre"].to_numpy() > pre_95
            pre_mask[:-1] += pre_mask[1:]
            deg_preds_filtered = copy.deepcopy(df_.loc[~pre_mask][targets].to_numpy())
            T_ref_filtered = copy.deepcopy(
                df_.loc[~pre_mask]["millisSinceGpsEpoch"].to_numpy()
            )
            deg_preds = scipy.interpolate.interp1d(
                T_ref_filtered,
                deg_preds_filtered,
                axis=0,
                bounds_error=None,
                fill_value="extrapolate",
                assume_sorted=True,
            )(df_["millisSinceGpsEpoch"].to_numpy())

            df_.loc[:, targets] = deg_preds
            dfs.append(df_)
        df = pd.concat(dfs, axis=0)

    else:
        ind = df[(df["dist_pro"] > pro_95) & (df["dist_pre"] > pre_95)][
            ["dist_pre", "dist_pro"]
        ].index
        # smoothing
        for i in ind:
            df.loc[i, "latDeg"] = (
                df.loc[i - 1, "latDeg"] + df.loc[i + 1, "latDeg"]
            ) / 2
            df.loc[i, "lngDeg"] = (
                df.loc[i - 1, "lngDeg"] + df.loc[i + 1, "lngDeg"]
            ) / 2
    return df


def filter_outlier_with_absloute(
    df: pd.DataFrame, max_velocity: float = 45.0, max_acc: float = 10.0
) -> pd.DataFrame:
    df["dist_pre"] = 0
    df["dist_pro"] = 0

    df["latDeg_pre"] = df["latDeg"].shift(periods=1, fill_value=0)
    df["lngDeg_pre"] = df["lngDeg"].shift(periods=1, fill_value=0)
    df["latDeg_pro"] = df["latDeg"].shift(periods=-1, fill_value=0)
    df["lngDeg_pro"] = df["lngDeg"].shift(periods=-1, fill_value=0)
    df["dist_pre"] = calc_haversine(df.latDeg_pre, df.lngDeg_pre, df.latDeg, df.lngDeg)
    df["dist_pro"] = calc_haversine(df.latDeg, df.lngDeg, df.latDeg_pro, df.lngDeg_pro)

    # start, end fix
    list_phone = df["phone"].unique()
    for phone in list_phone:
        ind_s = df[df["phone"] == phone].index[0]
        ind_e = df[df["phone"] == phone].index[-1]
        df.loc[ind_s, "dist_pre"] = 0
        df.loc[ind_e, "dist_pro"] = 0

    # 95% tile
    # pro_95 = df["dist_pro"].mean() + (df["dist_pro"].std() * 2)
    # pre_95 = df["dist_pre"].mean() + (df["dist_pre"].std() * 2)

    # find outlier data
    ind = df[(df["dist_pro"] > max_velocity) & (df["dist_pre"] > max_velocity)][
        ["dist_pre", "dist_pro"]
    ].index

    # smoothing
    for i in ind:
        df.loc[i, "latDeg"] = (df.loc[i - 1, "latDeg"] + df.loc[i + 1, "latDeg"]) / 2
        df.loc[i, "lngDeg"] = (df.loc[i - 1, "lngDeg"] + df.loc[i + 1, "lngDeg"]) / 2
    return df


def make_lerp_data(df: pd.DataFrame):
    """
    Generate interpolated lat,lng values for
    different phone times in the same collection.
    from https://www.kaggle.com/t88take/gsdc-phones-mean-prediction

    """
    org_columns = df.columns

    # Generate a combination of time x collection x phone and
    # combine it with the original data (generate records to be interpolated)
    assert (
        len(
            df[
                df.duplicated(
                    ["collectionName", "millisSinceGpsEpoch", "phoneName"], keep=False
                )
            ]
        )
        == 0
    )
    assert (
        len(df[df.duplicated(["collectionName", "millisSinceGpsEpoch"], keep=False)])
        > 0
    ), "there are multiple phone at the same obsevation"
    time_list = df[["collectionName", "millisSinceGpsEpoch"]].drop_duplicates()
    phone_list = df[["collectionName", "phoneName"]].drop_duplicates()

    # assert len(phone_list == 73), "all folders for phones equal 73"
    # each timestep row = # of unique phone
    tmp = time_list.merge(phone_list, on="collectionName", how="outer")

    # diffrent phone, eg. Pixel 4 and 4XLModded, has diffrente timestep,
    # so there are lots of nan after merge
    # and that's the target to be interpolated with the other available data.
    lerp_df = tmp.merge(
        df, on=["collectionName", "millisSinceGpsEpoch", "phoneName"], how="left"
    )
    lerp_df["phone"] = lerp_df["collectionName"] + "_" + lerp_df["phoneName"]
    lerp_df = lerp_df.sort_values(["phone", "millisSinceGpsEpoch"])

    # linear interpolation
    lerp_df["latDeg_prev"] = lerp_df["latDeg"].shift(1)
    lerp_df["latDeg_next"] = lerp_df["latDeg"].shift(-1)
    lerp_df["lngDeg_prev"] = lerp_df["lngDeg"].shift(1)
    lerp_df["lngDeg_next"] = lerp_df["lngDeg"].shift(-1)
    lerp_df["phone_prev"] = lerp_df["phone"].shift(1)
    lerp_df["phone_next"] = lerp_df["phone"].shift(-1)
    lerp_df["time_prev"] = lerp_df["millisSinceGpsEpoch"].shift(1)
    lerp_df["time_next"] = lerp_df["millisSinceGpsEpoch"].shift(-1)
    # Leave only records to be interpolated, nan & non_first, non_last data
    lerp_df = lerp_df[
        (lerp_df["latDeg"].isnull())
        & (lerp_df["phone"] == lerp_df["phone_prev"])
        & (lerp_df["phone"] == lerp_df["phone_next"])
    ].copy()
    # calc lerp, velocity x delta(time)
    lerp_df["latDeg"] = lerp_df["latDeg_prev"] + (
        (lerp_df["latDeg_next"] - lerp_df["latDeg_prev"])
        * (
            (lerp_df["millisSinceGpsEpoch"] - lerp_df["time_prev"])
            / (lerp_df["time_next"] - lerp_df["time_prev"])
        )
    )
    lerp_df["lngDeg"] = lerp_df["lngDeg_prev"] + (
        (lerp_df["lngDeg_next"] - lerp_df["lngDeg_prev"])
        * (
            (lerp_df["millisSinceGpsEpoch"] - lerp_df["time_prev"])
            / (lerp_df["time_next"] - lerp_df["time_prev"])
        )
    )

    # Leave only the data that has a complete set of previous and next data.
    lerp_df = lerp_df[~lerp_df["latDeg"].isnull()]

    return lerp_df[org_columns]


def calc_mean_pred(df: pd.DataFrame):
    """
    Make a prediction based on the average of the predictions of phones
    in the same collection.
    from https://www.kaggle.com/t88take/gsdc-phones-mean-prediction
    """
    lerp_df = make_lerp_data(df=df)
    add_lerp = pd.concat([df, lerp_df])
    # each time step == only one row, average over all phone latDeg,
    # lanDeg at each time step
    # eg. mean(original Deg Pixel4 and interpolated Deg 4XLModded with `make_lerp_data`)
    mean_pred_result = (
        add_lerp.groupby(["collectionName", "millisSinceGpsEpoch"])[
            ["latDeg", "lngDeg"]
        ]
        .mean()
        .reset_index()
    )
    base_cols = ["collectionName", "phoneName", "phone", "millisSinceGpsEpoch"]
    try:
        mean_pred_df = df[base_cols + ["latDeg_gt", "lngDeg_gt", "speedMps"]].copy()
    except Exception:
        mean_pred_df = df[base_cols].copy()
    mean_pred_df = mean_pred_df.merge(
        mean_pred_result[["collectionName", "millisSinceGpsEpoch", "latDeg", "lngDeg"]],
        on=["collectionName", "millisSinceGpsEpoch"],
        how="left",
    )
    return mean_pred_df


def get_removedevice(
    input_df: pd.DataFrame, divece: str = "SamsungS20Ultra"
) -> pd.DataFrame:
    """
    from
    https://www.kaggle.com/columbia2131/device-eda-interpolate-by-removing-device-en-ja
    """

    input_df["index"] = input_df.index
    input_df = input_df.sort_values("millisSinceGpsEpoch")
    input_df.index = input_df["millisSinceGpsEpoch"].values

    output_df = pd.DataFrame()
    for _, subdf in input_df.groupby("collectionName"):

        phones = subdf["phoneName"].unique()

        if (len(phones) == 1) or (divece not in phones):
            output_df = pd.concat([output_df, subdf])
            continue

        origin_df = subdf.copy()

        _index = subdf["phoneName"] == divece
        subdf.loc[_index, "latDeg"] = np.nan
        subdf.loc[_index, "lngDeg"] = np.nan
        subdf = subdf.interpolate(method="index", limit_area="inside")

        _index = subdf["latDeg"].isnull()
        subdf.loc[_index, "latDeg"] = origin_df.loc[_index, "latDeg"].values
        subdf.loc[_index, "lngDeg"] = origin_df.loc[_index, "lngDeg"].values

        output_df = pd.concat([output_df, subdf])

    output_df.index = output_df["index"].values
    output_df = output_df.sort_index()

    del output_df["index"]

    return output_df


def fft_filter_signal(signal: np.ndarray, threshold: float = 1e8) -> np.ndarray:
    orig_len = signal.shape[0]
    fourier = rfft(signal)
    frequencies = rfftfreq(signal.size, d=20e-3 / signal.size)
    fourier[frequencies > threshold] = 0
    filtered = irfft(fourier)
    reduced = orig_len - filtered.shape[0]
    if reduced > 0:
        filtered = np.concatenate([filtered] + [filtered[-reduced:]])

    return filtered


def apply_fft_filtering(
    df: pd.DataFrame, threshold: float = 1e8, targets: List[str] = ["latDeg", "lngDeg"]
) -> pd.DataFrame:
    unique_paths = df[["collectionName", "phoneName"]].drop_duplicates().to_numpy()
    for collection, phone in tqdm(unique_paths):
        cond = np.logical_and(
            df["collectionName"] == collection, df["phoneName"] == phone
        )
        for target in targets:
            df.loc[cond, target] = fft_filter_signal(
                signal=df.loc[cond, target].fillna(0).values, threshold=threshold
            )
    return df


def apply_gauss_smoothing(df, params):
    """
    from https://www.kaggle.com/bpetrb/adaptive-gauss-phone-mean
    """
    SZ_1 = params["sz_1"]
    SZ_2 = params["sz_2"]
    SZ_CRIT = params["sz_crit"]

    unique_paths = df[["collectionName", "phoneName"]].drop_duplicates().to_numpy()
    for collection, phone in unique_paths:
        cond = np.logical_and(
            df["collectionName"] == collection, df["phoneName"] == phone
        )
        data = df[cond][["latDeg", "lngDeg"]].to_numpy()

        lat_g1 = gaussian_filter1d(data[:, 0], np.sqrt(SZ_1))
        lon_g1 = gaussian_filter1d(data[:, 1], np.sqrt(SZ_1))
        lat_g2 = gaussian_filter1d(data[:, 0], np.sqrt(SZ_2))
        lon_g2 = gaussian_filter1d(data[:, 1], np.sqrt(SZ_2))

        lat_dif = data[1:, 0] - data[:-1, 0]
        lon_dif = data[1:, 1] - data[:-1, 1]

        lat_crit = np.append(
            np.abs(
                gaussian_filter1d(lat_dif, np.sqrt(SZ_CRIT))
                / (1e-9 + gaussian_filter1d(np.abs(lat_dif), np.sqrt(SZ_CRIT)))
            ),
            [0],
        )
        lon_crit = np.append(
            np.abs(
                gaussian_filter1d(lon_dif, np.sqrt(SZ_CRIT))
                / (1e-9 + gaussian_filter1d(np.abs(lon_dif), np.sqrt(SZ_CRIT)))
            ),
            [0],
        )

        df.loc[cond, "latDeg"] = lat_g1 * lat_crit + lat_g2 * (1.0 - lat_crit)
        df.loc[cond, "lngDeg"] = lon_g1 * lon_crit + lon_g2 * (1.0 - lon_crit)

    return df


def mean_with_other_phones(df):
    """
    from https://www.kaggle.com/bpetrb/adaptive-gauss-phone-mean
    """
    collections_list = df[["collectionName"]].drop_duplicates().to_numpy()

    for collection in collections_list:
        phone_list = (
            df[df["collectionName"].to_list() == collection][["phoneName"]]
            .drop_duplicates()
            .to_numpy()
        )

        phone_data = {}
        corrections = {}
        for phone in phone_list:
            cond = np.logical_and(
                df["collectionName"] == collection[0], df["phoneName"] == phone[0]
            ).to_list()
            phone_data[phone[0]] = df[cond][
                ["millisSinceGpsEpoch", "latDeg", "lngDeg"]
            ].to_numpy()

        for current in phone_data:
            correction = np.ones(phone_data[current].shape, dtype=np.float)
            correction[:, 1:] = phone_data[current][:, 1:]

            # Telephones data don't complitely match by time, so - interpolate.
            for other in phone_data:
                if other == current:
                    continue

                loc = interp1d(
                    phone_data[other][:, 0],
                    phone_data[other][:, 1:],
                    axis=0,
                    kind="linear",
                    copy=False,
                    bounds_error=None,
                    fill_value="extrapolate",
                    assume_sorted=True,
                )

                start_idx = 0
                stop_idx = 0
                for idx, val in enumerate(phone_data[current][:, 0]):
                    if val < phone_data[other][0, 0]:
                        start_idx = idx
                    if val < phone_data[other][-1, 0]:
                        stop_idx = idx

                if stop_idx - start_idx > 0:
                    correction[start_idx:stop_idx, 0] += 1
                    correction[start_idx:stop_idx, 1:] += loc(
                        phone_data[current][start_idx:stop_idx, 0]
                    )

            correction[:, 1] /= correction[:, 0]
            correction[:, 2] /= correction[:, 0]

            corrections[current] = correction.copy()

        for phone in phone_list:
            cond = np.logical_and(
                df["collectionName"] == collection[0], df["phoneName"] == phone[0]
            ).to_list()

            df.loc[cond, ["latDeg", "lngDeg"]] = corrections[phone[0]][:, 1:]

    return df


if __name__ == "__main__":
    pass
