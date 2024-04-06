import numpy as np
import mne
from mne import create_info
from mne.io import RawArray

def generate_raw_eeg_data(num_subjects, duration_sec, sampling_freq, n_channels=5):
    raw_eeg_data = []
    for _ in range(num_subjects):
        data = np.random.randn(n_channels, duration_sec * sampling_freq)
        ch_names = [f'EEG{i+1}' for i in range(n_channels)]
        ch_types = ['eeg'] * n_channels
        info = create_info(ch_names=ch_names, sfreq=sampling_freq, ch_types=ch_types)
        raw = RawArray(data, info)
        raw_eeg_data.append(raw)
    return raw_eeg_data

def generate_labels_for_frequent_thoughts(num_samples, fs, thought_frequency, thought_duration):
    labels = np.zeros(num_samples)
    for start in range(0, num_samples, int(fs * thought_frequency)):
        end = min(start + int(fs * thought_duration), num_samples)
        labels[start:end] = 1
    return labels

def add_labels_to_data(raw_eeg_datasets, thought_frequency, thought_duration, sfreq):
    for raw in raw_eeg_datasets:
        num_samples = raw.n_times
        labels = generate_labels_for_frequent_thoughts(num_samples, sfreq, thought_frequency, thought_duration)
        raw._data = np.vstack([raw.get_data(), labels])  # Добавляем лейблы как дополнительный канал

def add_correct_labels(raw_eeg_datasets, thought_frequency, thought_duration, sfreq):
    labeled_datasets = []
    for raw in raw_eeg_datasets:
        data = raw.get_data()
        num_samples = raw.n_times
        labels = generate_labels_for_frequent_thoughts(num_samples, sfreq, thought_frequency, thought_duration)
        labeled_data = np.vstack([data, labels])  # Добавляем лейблы как дополнительный канал
        ch_names = raw.ch_names + ['Labels']
        ch_types = raw.get_channel_types() + ['misc']
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        labeled_raw = mne.io.RawArray(labeled_data, info)
        labeled_datasets.append(labeled_raw)
    return labeled_datasets