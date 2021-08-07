from glob import glob
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import pandas as pd


def calc_stft(
    x: np.ndarray,
    n_fft: int = 256,
    hop_length: int = 1,
    win_length: int = 16,
    is_db: bool = False,
    is_pad: bool = False,
    amin: float = 1.0e-15,
    top_db: int = 200,
    ref: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    # need ref change
    length = x.shape[0]
    if is_pad:
        x = librosa.util.fix_length(x, length + n_fft // 2)
    # Short-time Fourier transform (STFT)
    D = librosa.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length,)

    # for reconstruction
    # np.abs(np.exp(1j * D_theta) * D_astfly - D).max()
    D_abs = np.abs(D)
    D_theta = np.angle(D)

    # Convert an amplitude spectrogram to Decibels-scaled spectrogram.
    if is_db:
        D_abs = librosa.amplitude_to_db(D_abs, ref=ref, amin=amin, top_db=top_db)
        D_theta = np.where(D_abs == D_abs.min(), 0.0, D_theta)
    return D_abs, D_theta


def calc_triangle_center(
    df: pd.DataFrame, targets: List[str] = ["latDeg", "lngDeg"], window_size: int = 3
) -> pd.DataFrame:
    """
        * - *   1 + 2 neighbors are make traiangle, we can create several aggregations.
      /
    *
    """

    dfs = []

    for phone_, df_ in df.groupby("phone"):
        tri_center = (
            df_[targets]
            .rolling(window=window_size, min_periods=2, center=True)
            .mean()
            .values
        )
        tri_center[0, :] = np.nan
        tri_center[-1, :] = np.nan
        for i, target in enumerate(targets):
            df_[target + "_left"] = df_[target].shift(1)
            df_[target + "_right"] = df_[target].shift(-1)
            df_[target + "_center"] = tri_center[:, i]
            df_[target + "_diff_center"] = df_[target] - tri_center[:, i]
            df_[target + "_diff_left"] = df_[target + "_left"] - tri_center[:, i]
            df_[target + "_diff_right"] = df_[target + "_right"] - tri_center[:, i]

        dfs.append(df_)

    return pd.concat(dfs)


def get_groundtruth(data_dir: Path) -> pd.DataFrame:
    """
    from
    https://www.kaggle.com/columbia2131/device-eda-interpolate-by-removing-device-en-ja
    """
    output_df = pd.DataFrame()

    for data_dir in glob(str(data_dir / "train/*/*/ground_truth.csv")):
        _df = pd.read_csv(data_dir)
        output_df = pd.concat([output_df, _df])
    output_df = output_df.reset_index(drop=True)

    _columns = ["latDeg", "lngDeg", "heightAboveWgs84EllipsoidM"]
    output_df[[col + "_gt" for col in _columns]] = output_df[_columns]
    output_df = output_df.drop(columns=_columns, axis=1)
    return output_df
