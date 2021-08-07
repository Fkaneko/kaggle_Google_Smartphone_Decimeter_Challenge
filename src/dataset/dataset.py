import copy
from typing import Any, Dict, List, Tuple

import albumentations as albu
import numpy as np
import pandas as pd
import torch

from src.dataset.utils import calc_stft


# from https://www.kaggle.com/hidehisaarai1213/pytorch-training-birdclef2021-starter
class WaveformDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sampling_list: pd.DataFrame,
        phone_sequences: Dict[str, Dict[str, np.ndarray]],
        stft_targets: List[str],
        stft_params: Dict[str, Any],
        input_width: int = 256,
        image_transforms: albu.Compose = None,
        is_test: bool = False,
        gt_as_mask: bool = False,
        period=20,
        rand_ratio: float = 0.0,
        rand_freq: float = 0.0,
        sigma: float = 5.0e-5,
    ):
        self.sampling_list = sampling_list
        self.phone_sequences = phone_sequences
        self.stft_targets = stft_targets
        self.stft_params = stft_params
        self.input_width = input_width
        self.is_test = is_test
        self.gt_as_mask = gt_as_mask
        self.image_transforms = image_transforms
        self.rand_ratio = rand_ratio
        self.rand_freq = rand_freq
        self.sigma = sigma
        if self.rand_freq > 0.0:
            print(
                '"xxx_diff_prev" inputs are assumed for the current wave augmentation'
            )

        self.period = period
        # self.validation = validation

        self.max = 0.0
        self.min = 255.0

        self.input_size = self.calc_stft_resize(
            input_width=self.input_width, n_fft=self.stft_params.n_fft
        )

    def __len__(self):
        return len(self.sampling_list)

    @staticmethod
    def handle_stft_normalize(
        img: np.ndarray,
        cnum: int = 3,
        is_encode: bool = False,
        is_db: bool = True,
        img_std: np.ndarray = None,
        img_mean: np.ndarray = None,
        gt_as_mask: bool = False,
        image_fomat: str = "matplot",
    ):
        acceptable_types = (np.ndarray, torch.Tensor)
        if is_encode:
            if is_db:
                img[..., :cnum] = -1.0 * img[..., :cnum]
            img[..., cnum:] = (img[..., cnum:] + 1.0) * 0.5 * 255.0
            if gt_as_mask:
                return img / 255.0
            else:
                return img
        else:
            if not (
                isinstance(img_mean, acceptable_types)
                or isinstance(img_std, acceptable_types)
            ):
                raise TypeError
            if img.dtype == np.float16:
                img = img.astype(np.float32)

            if not gt_as_mask:
                img = img * img_std + img_mean
            img *= 255.0
            if image_fomat == "matplot":
                if is_db:
                    D_abs = img[..., :cnum] * -1.0

                D_cos = img[..., cnum : cnum * 2] * 2.0 / 255.0 - 1.0
                D_sin = img[..., cnum * 2 : cnum * 3] * 2.0 / 255.0 - 1.0
            elif image_fomat == "torch":
                if is_db:
                    D_abs = img[:, :cnum] * -1.0
                D_cos = img[:, cnum : cnum * 2] * 2.0 / 255.0 - 1.0
                D_sin = img[:, cnum * 2 : cnum * 3] * 2.0 / 255.0 - 1.0

            return D_abs, D_cos, D_sin

    @staticmethod
    def get_image(
        phone_sequence: Dict[str, np.ndarray],
        stft_targets: List[str],
        stft_params: Dict[str, Any],
        get_gt: bool = False,
        gt_as_mask: bool = False,
    ) -> np.ndarray:
        D_abs, D_cos, D_sin = [], [], []
        if get_gt:
            stft_targets = [
                target.replace("_diff", "_gt_diff") for target in stft_targets
            ]

        for stft_target in stft_targets:
            abs_, theta_ = calc_stft(x=phone_sequence[stft_target], **stft_params,)
            D_abs.append(abs_)
            D_cos.append(np.cos(theta_))
            D_sin.append(np.sin(theta_))

        img = np.stack(D_abs + D_cos + D_sin, axis=-1)
        img = WaveformDataset.handle_stft_normalize(
            img=img,
            cnum=len(stft_targets),
            is_encode=True,
            is_db=stft_params["is_db"],
            gt_as_mask=gt_as_mask,
        )

        return img

    @staticmethod
    def calc_stft_resize(
        input_width: int = 256,
        n_fft: int = 256,
        base_size: int = 32,
        search_range: int = 1000,
    ) -> Tuple[int, int]:
        image_height = n_fft // 2 + 1
        image_width = input_width + 1

        for _ in range(search_range):
            if image_height % base_size == 0:
                break
            image_height += 1

        for _ in range(search_range):
            if image_width % base_size == 0:
                break
            image_width += 1

        return image_height, image_width

    def _range_check(self, img: np.ndarray) -> None:
        self.max = np.maximum(self.max, np.max(img[..., :3]))
        self.min = np.minimum(self.min, np.min(img[..., :3]))

    def __getitem__(self, idx: int):
        sample = self.sampling_list.loc[idx, :]
        phone = sample["phone"]
        phone_time = sample["phone_time"]
        phone_sequence = copy.deepcopy(self.phone_sequences[phone])

        # smaple
        for stft_target in self.stft_targets:
            phone_sequence[stft_target] = phone_sequence[stft_target][
                phone_time : phone_time + self.input_width
            ]

        if self.rand_freq > np.random.random():
            size = (len(self.stft_targets), int(self.input_width * self.rand_ratio))
            rand_sequence = np.zeros(
                (len(self.stft_targets), self.input_width), dtype=np.float64
            )
            rand_inds = np.random.choice(a=np.arange(self.input_width), size=size)
            rand_inds.sort()
            while np.any(rand_inds[:, 1:] == rand_inds[:, :-1]):
                rand_inds[:, 1:][rand_inds[:, 1:] == rand_inds[:, :-1]] += 1
            rand_inds = np.clip(rand_inds, 0, self.input_width - 1)
            rand_values = np.random.normal(loc=0.0, scale=self.sigma, size=size)
            for i, ind_ in enumerate(rand_inds):
                rand_sequence[i, rand_inds[i]] = rand_values[i]
                rand_sequence[i, 1:] -= rand_sequence[i, :-1]
            for j, stft_target in enumerate(self.stft_targets):
                phone_sequence[stft_target] += rand_sequence[j]

        img = self.get_image(
            phone_sequence=phone_sequence,
            stft_targets=self.stft_targets,
            stft_params=self.stft_params,
        )

        # self._range_check(img=img)
        if not self.is_test:
            for stft_target in self.stft_targets:
                stft_target = stft_target.replace("_diff", "_gt_diff")
                phone_sequence[stft_target] = phone_sequence[stft_target][
                    phone_time : phone_time + self.input_width
                ]

            target_image = self.get_image(
                phone_sequence=phone_sequence,
                stft_targets=self.stft_targets,
                stft_params=self.stft_params,
                get_gt=True,
                gt_as_mask=self.gt_as_mask,
            )
            augmented = self.image_transforms(image=img, target_image=target_image)
            target_image = augmented["target_image"].astype(np.float32)

            target_image = torch.from_numpy(target_image.transpose(2, 0, 1))
        else:
            augmented = self.image_transforms(image=img)
            target_image = torch.empty(0)

        img = augmented["image"]
        img = torch.from_numpy(img.transpose(2, 0, 1))
        return {
            "phone": phone,
            "phone_time": phone_time,
            "input_width": self.input_width,
            "millisSinceGpsEpoch": sample["millisSinceGpsEpoch"],
            "image": img,
            "target_image": target_image,
            **phone_sequence,
        }
