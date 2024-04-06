import pywt
import numpy as np
import mne
from mne import create_info
from mne.io import RawArray


def normalize_eeg_data(raw_eeg_datasets):
    normalized_datasets = []
    for raw in raw_eeg_datasets:
        data = raw.get_data()
        normalized_data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
        normalized_raw = RawArray(normalized_data, raw.info)
        normalized_datasets.append(normalized_raw)
    return normalized_datasets


def wavelet_transform(data, wavelet='db4', max_level=None):
    coeffs = pywt.wavedec(data, wavelet, level=max_level)
    return coeffs

def apply_third_level_wavelet(eeg_dataset, wavelet='db4', level=3):
    transformed_datasets = []
    for raw in eeg_dataset:
        transformed_data = []
        for i in range(len(raw.ch_names) - 1):  # Исключаем канал с метками
            data = raw.get_data(picks=[i])[0]
            coeffs = pywt.wavedec(data, wavelet, level=level)
            transformed_data.append(coeffs[2])  # Используем только коэффициенты третьего уровня
        transformed_datasets.append(np.array(transformed_data))
    return transformed_datasets