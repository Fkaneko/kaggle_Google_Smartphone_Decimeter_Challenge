import os
from typing import Dict, List, Optional, Tuple

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.dataset.utils import calc_stft, calc_triangle_center
from src.postprocess.metric import calc_haversine


def visualize_trafic(
    df: pd.DataFrame,
    center: Dict[str, float],
    zoom: int = 9,
    size: Optional[str] = None,
) -> None:
    df["second"] = (
        df["millisSinceGpsEpoch"] - df.iloc[0]["millisSinceGpsEpoch"]
    ) * 1.0e-3

    # Here, plotly detects color of series
    fig = px.scatter_mapbox(
        df,
        lat="latDeg",
        lon="lngDeg",
        color="phoneName",
        size=size,
        labels="second",
        zoom=zoom,
        center=center,
        height=600,
        width=800,
    )
    fig.update_layout(mapbox_style="stamen-terrain")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_layout(title_text="GPS trafic")
    fig.show()


def visualize_trafic_with_pred(
    df: pd.DataFrame,
    df_pred: Optional[pd.DataFrame],
    center: Dict[str, float],
    zoom: int = 9,
) -> None:
    """
    from
    https://www.kaggle.com/nayuts/let-s-visualize-dataset-to-understand
    """

    df["second"] = (
        df["millisSinceGpsEpoch"] - df["millisSinceGpsEpoch"].min()
    ) * 1.0e-3
    fig = go.Figure(
        go.Scattermapbox(
            mode="markers+lines",
            lon=df["lngDeg"],
            lat=df["latDeg"],
            text=df["second"],
            marker={"size": 10},
        )
    )
    if df_pred is not None:
        df_pred["second"] = (
            df_pred["millisSinceGpsEpoch"] - df_pred["millisSinceGpsEpoch"].min()
        ) * 1.0e-3
        fig.add_trace(
            go.Scattermapbox(
                mode="markers+lines",
                lon=df_pred["lngDeg"],
                lat=df_pred["latDeg"],
                text=df_pred["second"],
                marker={"size": 10},
            )
        )

    fig.update_layout(
        margin={"l": 0, "t": 0, "b": 0, "r": 0},
        mapbox={"center": center, "style": "stamen-terrain", "zoom": zoom},
    )
    fig.update_layout(mapbox_style="stamen-terrain")
    fig.update_layout(title_text="GPS trafic")
    fig.show()


def calc_azimuth_cos_sin(azimuth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cos = np.cos(azimuth)
    sin = np.sin(azimuth)
    return cos, sin


def calc_relative_path(
    df: pd.DataFrame, is_gt: bool = False, suffixes: List[str] = ["center", "left"]
) -> pd.DataFrame:
    def _left_to_right(n: int, delta_r: List[float]) -> float:
        if n == 0:
            return 0.0
        else:
            delta_r_n = delta_r[n]
            return _left_to_right(n - 1, delta_r=delta_r) + delta_r_n

    headers = ["latDeg_diff", "lngDeg_diff"]
    if is_gt:
        headers = [head.replace("diff", "gt_diff") for head in headers]

    diff_cols = []

    vector_dict = {"center": "r1", "right": "r2", "left": "r0"}

    for head in headers:
        for suffix in suffixes:
            diff_cols.append(["_".join([head, suffix]), vector_dict[suffix]])

    diffs = [col[0] for col in diff_cols]
    if suffixes == ["center", "left"]:
        new_dfs = []
        for phone, df_ in df.groupby("phone"):
            delta_r01_latDeg = (
                df_["latDeg_diff_center"] - df_["latDeg_diff_left"]
            ).values.tolist()
            delta_r01_lngDeg = (
                df_["lngDeg_diff_center"] - df_["lngDeg_diff_left"]
            ).values.tolist()

            delta_r12_latDeg = (
                df_["latDeg_diff_right"] - df_["latDeg_diff_center"]
            ).values.tolist()
            delta_r12_lngDeg = (
                df_["lngDeg_diff_right"] - df_["lngDeg_diff_center"]
            ).values.tolist()

            delta_r01_latDeg[-1] = delta_r12_latDeg[-2]
            delta_r01_lngDeg[-1] = delta_r12_lngDeg[-2]

            df_["rel_latDeg"] = [
                _left_to_right(n=n, delta_r=delta_r01_latDeg)
                for n in range(len(delta_r01_latDeg))
            ]
            df_["rel_lngDeg"] = [
                _left_to_right(n=n, delta_r=delta_r01_lngDeg)
                for n in range(len(delta_r01_lngDeg))
            ]

            new_dfs.append(df_)

    return df


def vis_stft(
    df: pd.DataFrame,
    phone: str,
    target: str = "lngDeg_diff_center",
    target_gt: Optional[str] = None,
    time_range: Optional[Tuple[int, int]] = None,
    is_db: bool = False,
    save_path: Optional[str] = None,
    win_length: int = 16,
    n_fft: int = 256,
    hop_length: int = 1,
    y_axis: str = "hz",
    amin: float = 1.0e-15,
    top_db: int = 200,
    ref: float = 1.0,
):

    x = df.query(f'phone == "{phone}"')[target].fillna(0.0).values
    if time_range is not None:
        x = x[time_range[0] : time_range[1]]
    D_abs, D_theta = calc_stft(
        x=x,
        win_length=win_length,
        is_db=is_db,
        n_fft=n_fft,
        hop_length=hop_length,
        amin=amin,
        top_db=top_db,
        ref=ref,
    )

    # === PLOT ===
    nrows = 4
    ncols = 2
    fig, ax = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(12, 6), sharey=False, sharex=False,
    )
    fig.suptitle("Log Frequency Spectrogram", fontsize=16)
    # fig.delaxes(ax[1, 2])

    title_text = f"W:{win_length}, T:[{time_range[0]}:{time_range[1]}]"
    if is_db:
        title_text = "Axis:dB, " + title_text
    else:
        title_text = "Axis:Lin, " + title_text
    ax[0][0].set_title(target + ", " + title_text, fontsize=10)
    ax[0][1].set_title(phone, fontsize=10)

    if time_range is not None:
        err = df.iloc[time_range[0] : time_range[1]]
    else:
        err = df
    ax[0][0].plot(err["millisSinceGpsEpoch"], err[target], label=f"{target}")
    if target_gt is not None:
        ax[0][0].plot(err["millisSinceGpsEpoch"], err[target_gt], label=f"{target_gt}")
        ax[0][1].plot(
            err["millisSinceGpsEpoch"], err["dist_err"], label="err(baseline)"
        )
        ax[0][1].set_xlim(
            err["millisSinceGpsEpoch"].min(), err["millisSinceGpsEpoch"].max()
        )
    ax[0][0].legend(loc="upper right")
    ax[0][1].legend(loc="upper right")
    ax[0][0].set_xlim(
        err["millisSinceGpsEpoch"].min(), err["millisSinceGpsEpoch"].max()
    )

    for nrow, mat in enumerate([D_abs, np.cos(D_theta), np.sin(D_theta)]):

        img = librosa.display.specshow(
            mat,
            sr=1,
            hop_length=hop_length,
            x_axis="time",
            y_axis=y_axis,
            cmap="cool",
            ax=ax[nrow + 1][0],
        )
        plt.colorbar(img, ax=ax[nrow + 1][0])

    plt.colorbar(img, ax=ax[0][0])
    plt.colorbar(img, ax=ax[0][1])

    if target_gt is not None:
        x_gt = df.query(f'phone == "{phone}"')[target_gt].fillna(0.0).values
        if time_range is not None:
            x_gt = x_gt[time_range[0] : time_range[1]]
        D_abs_gt, D_theta_gt = calc_stft(
            x=x_gt,
            win_length=win_length,
            is_db=is_db,
            n_fft=n_fft,
            hop_length=hop_length,
            is_pad=False,
            amin=amin,
            top_db=top_db,
            ref=ref,
        )
        for nrow, mat in enumerate([D_abs_gt, np.cos(D_theta_gt), np.sin(D_theta_gt)]):
            img = librosa.display.specshow(
                mat,
                sr=1,
                hop_length=hop_length,
                x_axis="time",
                y_axis=y_axis,
                cmap="cool",
                ax=ax[nrow + 1][1],
            )
            # ax.set_title("ASTFLY", fontsize=13)
            plt.colorbar(img, ax=ax[nrow + 1][1])

    if save_path is not None:
        suffix = ".png"
        if time_range is not None:
            suffix = f"time_range{time_range[0]}:{time_range[1]}" + suffix
        suffix = f"win_{win_length}_" + suffix
        if is_db:
            suffix = "_dB_" + suffix
        else:
            suffix = "_Lin_" + suffix

        save_path = save_path.replace(".png", suffix)
        plt.savefig(save_path, dpi=300)
        plt.close()

    if target_gt is not None:
        plot_rec(
            x_gt=x_gt,
            D_abs=D_abs,
            D_theta=D_theta,
            D_abs_gt=D_abs_gt,
            D_theta_gt=D_theta_gt,
            length=x.shape[0],
            is_db=is_db,
            hop_length=hop_length,
            win_length=win_length,
            save_path=save_path,
        )


def plot_rec(
    x_gt: np.ndarray,
    D_abs: np.ndarray,
    D_theta: np.ndarray,
    D_abs_gt: np.ndarray,
    D_theta_gt: np.ndarray,
    length: int = 256,
    is_db: bool = True,
    hop_length: int = 1,
    win_length: int = 16,
    save_path: Optional[str] = None,
    logger: Optional = None,
    log_name: Optional[str] = None,
    logger_name: str = "tensorboard",
    global_step: int = 0,
    x: Optional[np.ndarray] = None,
    target_name: Optional[str] = None,
) -> None:

    if is_db:
        D_abs = librosa.db_to_amplitude(D_abs, ref=1.0)
        D_abs_gt = librosa.db_to_amplitude(D_abs_gt, ref=1.0)

    x_rec = librosa.istft(
        np.exp(1j * D_theta) * D_abs,
        hop_length=hop_length,
        win_length=win_length,
        length=length,
    )

    # x_rec_with_gt_abs = librosa.istft(
    #     np.exp(1j * D_theta) * D_abs_gt,
    #     hop_length=hop_length,
    #     win_length=win_length,
    #     length=length,
    # )
    # x_rec_with_gt_theta = librosa.istft(
    #     np.exp(1j * D_theta_gt) * D_abs,
    #     hop_length=hop_length,
    #     win_length=win_length,
    #     length=length,
    # )
    x_rec_with_gt = librosa.istft(
        np.exp(1j * D_theta_gt) * D_abs_gt,
        hop_length=hop_length,
        win_length=win_length,
        length=length,
    )

    # === PLOT ===
    nrows = 1
    ncols = 1
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(12, 6), sharey=True, sharex=True,
    )
    axes.plot(np.abs(x_rec - x_gt), label="pred")
    # axes.plot(np.abs(x_rec_with_gt_abs - x_gt), label="with_gt_abs")
    # axes.plot(np.abs(x_rec_with_gt_theta - x_gt), label="with_gt_theta")
    axes.plot(np.abs(x_rec_with_gt - x_gt), label="with_gt")
    if x is not None:
        axes.plot(np.abs(x - x_gt), label="pred-baseline")
    rec_err = np.max(np.abs(x_rec_with_gt - x_gt))
    axes.set_yscale("log")
    axes.set_ylabel("mean abs error")
    axes.set_xlabel("time step")
    axes.grid()
    title = "stft reconstruction check"
    if target_name is not None:
        title = target_name + ": " + title
    fig.suptitle(title)
    plt.legend()

    if save_path is not None:
        recon_dir = os.path.join(os.path.dirname(save_path), "recon")
        print(f"{recon_dir}: {rec_err:4.3e}")
        os.makedirs(recon_dir, exist_ok=True)
        save_path = os.path.join(recon_dir, os.path.basename(save_path))
        plt.savefig(save_path, dpi=300)

    if logger is not None:
        axes.set_ylim(1.0e-9, 1.0e-4)
        if logger_name == "tensorboard":
            logger.experiment.add_figure(
                "sequernce_prediction", fig, global_step=global_step,
            )
        elif logger_name == "neptune":
            logger.experiment[log_name].log(fig)

    plt.close()


def add_distance_diff(df: pd.DataFrame, is_test: bool = False) -> pd.DataFrame:
    def _get_cols(is_test: bool, suffix: str = "prev") -> List[str]:
        headers = [
            "latDeg",
            "lngDeg",
            "latDeg_diff",
            "lngDeg_diff",
            "dist",
            "azimuth",
            "cos_azimuth",
            "sin_azimuth",
            "dist_center",
            "azimuth_center",
        ]
        train_col = [
            "latDeg_gt",
            "lngDeg_gt",
            "dist_gt",
            "latDeg_gt_diff",
            "lngDeg_gt_diff",
            "azimuth_gt",
            "cos_azimuth_gt",
            "sin_azimuth_gt",
            "dist_gt_center",
            "azimuth_gt_center",
        ]

        cols = []
        if not is_test:
            headers = headers + train_col

        for header in headers:
            cols.append(header + "_" + suffix)

        return cols

    df["latDeg_prev"] = df["latDeg"].shift(1)
    df["latDeg_next"] = df["latDeg"].shift(-1)
    df["lngDeg_prev"] = df["lngDeg"].shift(1)
    df["lngDeg_next"] = df["lngDeg"].shift(-1)
    df["phone_prev"] = df["phone"].shift(1)
    df["phone_next"] = df["phone"].shift(-1)

    df["latDeg_diff_prev"] = df["latDeg"] - df["latDeg_prev"]
    df["latDeg_diff_next"] = df["latDeg_next"] - df["latDeg"]
    df["lngDeg_diff_prev"] = df["lngDeg"] - df["lngDeg_prev"]
    df["lngDeg_diff_next"] = df["lngDeg_next"] - df["lngDeg"]

    df["dist_prev"], df["azimuth_prev"] = calc_haversine(
        df["latDeg"],
        df["lngDeg"],
        df["latDeg_prev"],
        df["lngDeg_prev"],
        return_azimuth=True,
    )
    df["cos_azimuth_prev"], df["sin_azimuth_prev"] = calc_azimuth_cos_sin(
        azimuth=df["azimuth_prev"].values
    )
    df["dist_next"], df["azimuth_next"] = calc_haversine(
        df["latDeg"],
        df["lngDeg"],
        df["latDeg_next"],
        df["lngDeg_next"],
        return_azimuth=True,
    )
    df["cos_azimuth_next"], df["sin_azimuth_next"] = calc_azimuth_cos_sin(
        azimuth=df["azimuth_next"].values
    )

    df = calc_triangle_center(df=df, targets=["latDeg", "lngDeg"])

    df["dist_center"], df["azimuth_center"] = calc_haversine(
        df["latDeg_center"],
        df["lngDeg_center"],
        df["latDeg"],
        df["lngDeg"],
        return_azimuth=True,
    )
    df["cos_azimuth_center"], df["sin_azimuth_center"] = calc_azimuth_cos_sin(
        azimuth=df["azimuth_center"].values
    )

    if not is_test:
        df["latDeg_gt_prev"] = df["latDeg_gt"].shift(1)
        df["latDeg_gt_next"] = df["latDeg_gt"].shift(-1)
        df["lngDeg_gt_prev"] = df["lngDeg_gt"].shift(1)
        df["lngDeg_gt_next"] = df["lngDeg_gt"].shift(-1)

        df["latDeg_gt_diff_prev"] = df["latDeg_gt"] - df["latDeg_gt_prev"]
        df["latDeg_gt_diff_next"] = df["latDeg_gt_next"] - df["latDeg_gt"]
        df["lngDeg_gt_diff_prev"] = df["lngDeg_gt"] - df["lngDeg_gt_prev"]
        df["lngDeg_gt_diff_next"] = df["lngDeg_gt_next"] - df["lngDeg_gt"]

        df["latDeg_gt_prev"] = df["latDeg_gt"].shift(1)
        df["latDeg_gt_next"] = df["latDeg_gt"].shift(-1)
        df["lngDeg_gt_prev"] = df["lngDeg_gt"].shift(1)
        df["lngDeg_gt_next"] = df["lngDeg_gt"].shift(-1)

        df["dist_gt_prev"], df["azimuth_gt_prev"] = calc_haversine(
            df["latDeg_gt"],
            df["lngDeg_gt"],
            df["latDeg_gt_prev"],
            df["lngDeg_gt_prev"],
            return_azimuth=True,
        )
        df["cos_azimuth_gt_prev"], df["sin_azimuth_gt_prev"] = calc_azimuth_cos_sin(
            azimuth=df["azimuth_gt_prev"].values
        )
        df["dist_gt_next"], df["azimuth_gt_next"] = calc_haversine(
            df["latDeg_gt"],
            df["lngDeg_gt"],
            df["latDeg_gt_next"],
            df["lngDeg_gt_next"],
            return_azimuth=True,
        )

        df["cos_azimuth_gt_next"], df["sin_azimuth_gt_next"] = calc_azimuth_cos_sin(
            azimuth=df["azimuth_gt_next"].values
        )

        df = calc_triangle_center(df=df, targets=["latDeg_gt", "lngDeg_gt"])

        df["dist_gt_center"], df["azimuth_gt_center"] = calc_haversine(
            df["latDeg_gt_center"],
            df["lngDeg_gt_center"],
            df["latDeg_gt"],
            df["lngDeg_gt"],
            return_azimuth=True,
        )
        df["cos_azimuth_gt_center"], df["sin_azimuth_gt_center"] = calc_azimuth_cos_sin(
            azimuth=df["azimuth_gt_center"].values
        )

    df.loc[
        df["phone"] != df["phone_prev"], _get_cols(is_test=is_test, suffix="prev")
    ] = np.nan

    df.loc[
        df["phone"] != df["phone_next"], _get_cols(is_test=is_test, suffix="next")
    ] = np.nan

    return df


def visualize_err_move_dist(
    df: pd.DataFrame,
    phone: str,
    reject_outlier: bool = True,
    is_test: bool = False,
    skip_calc_diff: bool = False,
):
    """
    visualize baseline error and relative move distance
    modify from
    https://www.kaggle.com/t88take/gsdc-eda-error-when-stopping
    """
    if not skip_calc_diff:
        df = add_distance_diff(df=df, is_test=is_test)

    fig, axes = plt.subplots(figsize=(20, 10), nrows=3, sharex=True)
    df = df[df["phone"] == phone]

    axes[1].plot(
        df["millisSinceGpsEpoch"],
        df["dist_prev"],
        # df["dist_center"],
        # df["latDeg_diff_prev"],
        # df["lngDeg_diff_prev"],
        # df["latDeg_diff_center"],
        # df["lngDeg_diff_center"],
        # df["azimuth_center"],
        label="move dist(baseline)"
        # df["millisSinceGpsEpoch"], df["latDeg_prev"], label="move dist(baseline)"
        # df["millisSinceGpsEpoch"], df["lngDeg_prev"] - df["lngDeg_prev"].mean(), label="move dist(baseline)"
    )
    # axes[2].plot(
    #     df["millisSinceGpsEpoch"], df["azimuth_prev"], label="move azimuth(baseline)",
    # )
    axes[2].plot(
        df["millisSinceGpsEpoch"],
        df["cos_azimuth_prev"],
        label="move cos azimuth(baseline)",
    )
    axes[2].plot(
        df["millisSinceGpsEpoch"],
        df["sin_azimuth_prev"],
        label="move sin azimuth(baseline)",
    )

    if not is_test:
        df["dist_err"] = calc_haversine(
            df["latDeg_gt"], df["lngDeg_gt"], df["latDeg"], df["lngDeg"],
        )
        if reject_outlier:
            th = (df["dist_err"].std() * 3) + df["dist_err"].mean()
            df = df[df["dist_err"] < th]
        axes[0].plot(df["millisSinceGpsEpoch"], df["dist_err"], label="err(baseline)")
        # axes[1].plot(df["millisSinceGpsEpoch"], df["speedMps"], label="speedMps")
        axes[1].plot(
            df["millisSinceGpsEpoch"],
            df["dist_gt_prev"],
            # df["dist_gt_center"],
            # df["latDeg_gt_diff_prev"],
            # df["latDeg_gt_diff_center"],
            # df["lngDeg_gt_diff_center"],
            # df["azimuth_gt_center"],
            # df["dist_gt_prev"] - df["dist_gt_prev"].mean(),
            # df["lngDeg_gt_prev"] - df["lngDeg_gt_prev"].mean(),
            label="move dist(ground_truth)",
        )
        # axes[2].plot(
        #     df["millisSinceGpsEpoch"],
        #     df["azimuth_gt_prev"],
        #     label="move azimuth(ground_truth)",
        # )
        axes[2].plot(
            df["millisSinceGpsEpoch"],
            df["cos_azimuth_gt_prev"],
            label="move cos azimuth(ground_truth)",
        )
        axes[2].plot(
            df["millisSinceGpsEpoch"],
            df["sin_azimuth_gt_prev"],
            label="move sin azimuth(ground_truth)",
        )

    axes[0].legend(loc="upper right")
    axes[1].legend(loc="upper right")
    axes[2].legend(loc="upper right")
    axes[0].grid(color="g", linestyle=":", linewidth=0.3)
    axes[1].grid(color="g", linestyle=":", linewidth=0.3)
    axes[2].grid(color="g", linestyle=":", linewidth=0.3)

    fig.suptitle(phone, fontsize=16)
