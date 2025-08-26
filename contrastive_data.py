import torch
from torch.utils.data import Dataset
import numpy as np


class ContrastiveSignalDataset(Dataset):

    def __init__(self, emitters, max_length=300, norm_mode="physical"):
        self.emitters = emitters
        self.n_samples = len(emitters)
        self.max_length = max_length
        self.norm_mode = norm_mode

    def normalize_signal(self, values):
        values = values[:, [0, 1, 2, 4]]  # freq, pw, bw, pri

        if self.norm_mode == "physical":
            fc_min, fc_max = 3.5e9, 19e9
            pri_min, pri_max = 10e-6, 600e-6
            pw_min, pw_max = 2e-7, 1e-4
            bw_min, bw_max = 1 / pw_max, 1000 / pri_min

            # Uppdaterad scaling till intervallet [-1, 1]
            values[:, 0] = 2 * ((values[:, 0] - fc_min) / (fc_max - fc_min)) - 1
            values[:, 1] = 2 * ((values[:, 1] - pw_min) / (pw_max - pw_min)) - 1
            values[:, 2] = 2 * ((values[:, 2] - bw_min) / (bw_max - bw_min)) - 1
            values[:, 3] = 2 * ((values[:, 3] - pri_min) / (pri_max - pri_min)) - 1

        elif self.norm_mode == "zscore":
            mean = values.mean(axis=0)
            std = values.std(axis=0)
            std[std == 0] = 1
            values = (values - mean) / std

        elif self.norm_mode == "minmax":
            min_vals = values.min(axis=0)
            max_vals = values.max(axis=0)
            scale = max_vals - min_vals
            scale[scale == 0] = 1
            values = (values - min_vals) / scale

        elif self.norm_mode == "none":
            pass
        else:
            raise ValueError(f"Unknown normalization mode: {self.norm_mode}")

        return values

    def pad_or_clip(self, signal):
        values = signal.values.astype(np.float32)
        values = self.normalize_signal(values)

        n_pulses, n_features = values.shape
        mask = np.zeros(self.max_length, dtype=bool)

        if n_pulses < self.max_length:
            pad_width = self.max_length - n_pulses
            pad = np.zeros((pad_width, n_features), dtype=np.float32)
            values = np.vstack([values, pad])
            mask[n_pulses:] = True  # True = padding
        elif n_pulses > self.max_length:
            values = values[: self.max_length]
            mask[:] = False
        else:
            mask[:] = False

        return torch.tensor(values, dtype=torch.float32), torch.tensor(
            mask, dtype=torch.bool
        )

    def __getitem__(self, idx):
        emitter = np.random.choice(self.emitters)

        signal_clean = emitter.signal(noise=False)
        signal_noise = emitter.signal(noise=True)

        x_clean, mask_clean = self.pad_or_clip(signal_clean)
        x_noise, mask_noise = self.pad_or_clip(signal_noise)

        return x_clean, mask_clean, x_noise, mask_noise, emitter.id

    def __len__(self):
        return self.n_samples
