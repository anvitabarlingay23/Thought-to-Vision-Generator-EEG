import json
import os
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

# --- CONFIGURATION ---
SAMPLE_RATE = 250  # Hz
LOWCUT = 1.0
HIGHCUT = 40.0
NOTCH_FREQ = 50.0
ARTIFACT_THRESHOLD_UV = 100.0

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def notch_filter(data, notch_freq, fs):
    nyq = 0.5 * fs
    q = 30.0
    b, a = iirnotch(notch_freq / nyq, q)
    return filtfilt(b, a, data)

def preprocess_subject(subject_id, session_output_root="session_output"):
    print(f"\n--- Preprocessing Subject: {subject_id} ---")
    subject_dir = os.path.join(session_output_root, subject_id)
    input_filepath = os.path.join(subject_dir, f"{subject_id}_combined_two_channel_microvolts.json")
    output_filepath = os.path.join(subject_dir, f"{subject_id}_filtered_eeg.json")

    if not os.path.exists(input_filepath):
        print(f"  > Error: Input file not found. Skipping {subject_id}.")
        return

    with open(input_filepath, 'r') as f:
        all_trials = json.load(f)

    clean_trials = []
    rejected = 0
    total = len(all_trials)
    print(f"  > Found {total} raw trials.")

    for trial in all_trials:
        eeg_ch1 = np.array(trial['eeg_data'][0])
        eeg_ch2 = np.array(trial['eeg_data'][1])

        # 1. Artifact rejection
        if np.any(np.abs(eeg_ch1) > ARTIFACT_THRESHOLD_UV) or np.any(np.abs(eeg_ch2) > ARTIFACT_THRESHOLD_UV):
            rejected += 1
            continue

        # 2. Filters
        filtered_ch1 = butter_bandpass_filter(notch_filter(eeg_ch1, NOTCH_FREQ, SAMPLE_RATE), LOWCUT, HIGHCUT, SAMPLE_RATE)
        filtered_ch2 = butter_bandpass_filter(notch_filter(eeg_ch2, NOTCH_FREQ, SAMPLE_RATE), LOWCUT, HIGHCUT, SAMPLE_RATE)

        trial['eeg_data'] = [filtered_ch1.tolist(), filtered_ch2.tolist()]
        clean_trials.append(trial)

    with open(output_filepath, 'w') as f:
        json.dump(clean_trials, f, indent=2)

    print(f"  > Completed {subject_id}: {len(clean_trials)} accepted, {rejected} rejected.")
    print(f"  > Saved to: {output_filepath}")

def main():
    session_output_root = "session_output"
    if not os.path.isdir(session_output_root):
        raise SystemExit(f"Error: '{session_output_root}' not found.")

    subject_ids = [d for d in os.listdir(session_output_root) if os.path.isdir(os.path.join(session_output_root, d))]

    if not subject_ids:
        print("No subject folders found.")
        return

    print("Available subjects:")
    for i, s in enumerate(subject_ids):
        print(f"  {i+1}. {s}")

    selected = input("\nEnter the subject IDs or numbers to process (comma-separated, or 'all'): ").strip()

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

    print(f"\nProcessing {len(chosen_subjects)} subject(s): {chosen_subjects}")
    for subject_id in chosen_subjects:
        preprocess_subject(subject_id, session_output_root)

    print("\n✅ Preprocessing complete.")

if __name__ == "__main__":
    main()
