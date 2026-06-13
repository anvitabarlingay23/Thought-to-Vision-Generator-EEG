import json
import os
import numpy as np
import pywt

# --- CONFIGURATION ---
WAVELET = 'db4'
WAVELET_LEVELS = 4
SAMPLE_RATE = 250

def get_fft_features(epoch_data, sample_rate):
    n = len(epoch_data)
    fft_vals = np.fft.fft(epoch_data)
    fft_freq = np.fft.fftfreq(n, 1/sample_rate)

    fft_vals = np.abs(fft_vals[fft_freq >= 0])
    fft_freq = fft_freq[fft_freq >= 0]

    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 40)
    }

    band_powers = []
    for band in bands.values():
        freq_ix = np.where((fft_freq >= band[0]) & (fft_freq <= band[1]))[0]
        band_powers.append(np.mean(fft_vals[freq_ix]))
    return np.array(band_powers)

def get_wavelet_features(epoch_data, wavelet, levels):
    coeffs = pywt.wavedec(epoch_data, wavelet, level=levels)
    features = []
    for c in coeffs:
        features.extend([np.mean(c), np.std(c), np.median(np.abs(c))])
    return np.array(features)

def analyze_subject(subject_id, session_output_root="session_output"):
    print(f"\n--- Analyzing Subject: {subject_id} ---")
    subject_dir = os.path.join(session_output_root, subject_id)
    input_filepath = os.path.join(subject_dir, f"{subject_id}_filtered_eeg.json")
    output_filepath = os.path.join(subject_dir, f"{subject_id}_processed_features.json")

    if not os.path.exists(input_filepath):
        print(f"  > Error: Filtered file not found for {subject_id}. Run preprocessing first.")
        return

    with open(input_filepath, 'r') as f:
        clean_trials = json.load(f)

    feature_data = []
    print(f"  > Found {len(clean_trials)} clean trials to analyze.")

    for trial in clean_trials:
        eeg_ch1 = np.array(trial['eeg_data'][0])
        eeg_ch2 = np.array(trial['eeg_data'][1])

        fft_ch1 = get_fft_features(eeg_ch1, SAMPLE_RATE)
        fft_ch2 = get_fft_features(eeg_ch2, SAMPLE_RATE)

        wavelet_ch1 = get_wavelet_features(eeg_ch1, WAVELET, WAVELET_LEVELS)
        wavelet_ch2 = get_wavelet_features(eeg_ch2, WAVELET, WAVELET_LEVELS)

        combined_ch1 = np.concatenate([fft_ch1, wavelet_ch1])
        combined_ch2 = np.concatenate([fft_ch2, wavelet_ch2])

        feature_data.append({
            "subject": trial["subject"],
            "label": trial["label"],
            "image": trial["image"],
            "features": [combined_ch1.tolist(), combined_ch2.tolist()]
        })

    with open(output_filepath, 'w') as f:
        json.dump(feature_data, f, indent=2)

    print(f"  > Features saved to: {output_filepath}")

def main():
    session_output_root = "session_output"
    if not os.path.isdir(session_output_root):
        raise SystemExit(f"Error: '{session_output_root}' not found.")

    subject_ids = [d for d in os.listdir(session_output_root) if os.path.isdir(os.path.join(session_output_root, d))]

    if not subject_ids:
        print("No subjects found.")
        return

    print("Available subjects:")
    for i, s in enumerate(subject_ids):
        print(f"  {i+1}. {s}")

    selected = input("\nEnter the subject IDs or numbers to analyze (comma-separated, or 'all'): ").strip()

    if selected.lower() == 'all':
        chosen_subjects = subject_ids
    else:
        indices = [s.strip() for s in selected.split(',')]
        chosen_subjects = []
        for idx in indices:
            if idx.isdigit() and 1 <= int(idx) <= len(subject_ids):
                chosen_subjects.append(subject_ids[int(idx)-1])
            elif idx in subject_ids:
                chosen_subjects.append(idx)
            else:
                print(f"  ⚠️ Skipping invalid entry: {idx}")

    print(f"\nAnalyzing {len(chosen_subjects)} subject(s): {chosen_subjects}")
    for subject_id in chosen_subjects:
        analyze_subject(subject_id, session_output_root)

    print("\n✅ Feature extraction complete.")

if __name__ == "__main__":
    main()
