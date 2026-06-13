import json
import os
import numpy as np
import pywt

# --- CONFIGURATION ---
WAVELET = 'db4'    # Daubechies 4 is a robust choice for neuro-electrical signals
WAVELET_LEVELS = 4 # Decompose the signal into this many frequency bands

def get_fft_features(epoch_data, sample_rate):
    """
    Performs a Fast Fourier Transform and returns the power in key brainwave bands.
    This provides a summary of 'what' frequencies were active.
    """
    n = len(epoch_data)
    fft_vals = np.fft.fft(epoch_data)
    fft_freq = np.fft.fftfreq(n, 1/sample_rate)
    
    # Take only the positive frequencies
    fft_vals = np.abs(fft_vals[fft_freq >= 0])
    fft_freq = fft_freq[fft_freq >= 0]
    
    # Define EEG bands
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 40) # Capped at 40Hz by our bandpass filter
    }
    
    band_powers = []
    for band in bands.values():
        freq_ix = np.where((fft_freq >= band[0]) & (fft_freq <= band[1]))[0]
        band_power = np.mean(fft_vals[freq_ix])
        band_powers.append(band_power)
        
    return np.array(band_powers)

def get_wavelet_features(epoch_data, wavelet, levels):
    """
    Performs a multilevel wavelet decomposition.
    This provides a detailed view of 'when' different frequency events occurred.
    We'll extract statistical features from the wavelet coefficients.
    """
    coeffs = pywt.wavedec(epoch_data, wavelet, level=levels)
    features = []
    for c in coeffs:
        features.extend([np.mean(c), np.std(c), np.median(np.abs(c))])
    return np.array(features)

def analyze_subject(subject_id, session_output_root="session_output"):
    """
    Loads a subject's filtered data, performs analysis (FFT, Wavelet),
    and saves the final feature vectors.
    """
    print(f"\n--- Analyzing Subject: {subject_id} ---")
    subject_dir = os.path.join(session_output_root, subject_id)
    input_filepath = os.path.join(subject_dir, f"{subject_id}_filtered_eeg.json")
    output_filepath = os.path.join(subject_dir, f"{subject_id}_processed_features.json")

    if not os.path.exists(input_filepath):
        print(f"  > Error: Filtered file not found. Please run preprocessing first for {subject_id}.")
        return

    with open(input_filepath, 'r') as f:
        clean_trials = json.load(f)

    feature_data = []
    print(f"  > Found {len(clean_trials)} clean trials to analyze.")

    for trial in clean_trials:
        eeg_ch1 = np.array(trial['eeg_data'][0])
        eeg_ch2 = np.array(trial['eeg_data'][1])
        
        # 1. Fourier Transform Analysis
        fft_features_ch1 = get_fft_features(eeg_ch1, 250)
        fft_features_ch2 = get_fft_features(eeg_ch2, 250)

        # 2. Wavelet Transform Analysis
        wavelet_features_ch1 = get_wavelet_features(eeg_ch1, WAVELET, WAVELET_LEVELS)
        wavelet_features_ch2 = get_wavelet_features(eeg_ch2, WAVELET, WAVELET_LEVELS)

        # 3. Combine all features into a single vector for the LSTM
        combined_features_ch1 = np.concatenate([fft_features_ch1, wavelet_features_ch1])
        combined_features_ch2 = np.concatenate([fft_features_ch2, wavelet_features_ch2])

        feature_trial = {
            "subject": trial["subject"],
            "label": trial["label"],
            "image": trial["image"],
            "features": [
                combined_features_ch1.tolist(),
                combined_features_ch2.tolist()
            ]
        }
        feature_data.append(feature_trial)

    with open(output_filepath, 'w') as f:
        json.dump(feature_data, f, indent=2)

    print(f"  > Analysis complete.")
    print(f"  > Final feature vectors saved to: {output_filepath}")


def main():
    session_output_root = "session_output"
    if not os.path.isdir(session_output_root):
        raise SystemExit(f"Error: Root directory '{session_output_root}' not found.")
        
    subject_ids = [d for d in os.listdir(session_output_root) if os.path.isdir(os.path.join(session_output_root, d))]
    
    if not subject_ids:
        print("No subject folders found to analyze.")
        return
        
    print(f"Starting analysis for {len(subject_ids)} subjects...")
    for subject_id in subject_ids:
        analyze_subject(subject_id, session_output_root)
    print("\nAll subjects have been analyzed.")

if __name__ == "__main__":
    # Ensure you have the necessary libraries: pip install numpy pywavelets
    main()
