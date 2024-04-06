from eeg_data_generation.data_generation_and_labeling import generate_raw_eeg_data, add_correct_labels
from eeg_data_generation.transformation import normalize_eeg_data, apply_third_level_wavelet

def main():
    num_subjects = 5
    duration_sec = 30
    sampling_freq = 100
    n_channels = 5

    raw_eeg_data = generate_raw_eeg_data(num_subjects, duration_sec, sampling_freq, n_channels)

    thought_frequency = 5
    thought_duration = 1

    labeled_raw_eeg_data = add_correct_labels(raw_eeg_data, thought_frequency, thought_duration, sampling_freq)

    normalized_raw_eeg_data = normalize_eeg_data(labeled_raw_eeg_data)

    third_level_wavelet_data = apply_third_level_wavelet(normalized_raw_eeg_data, wavelet='db4', level=3)

    print(f"Shape of transformed data for the first subject: {third_level_wavelet_data[0].shape}")

if __name__ == "__main__":
    main()