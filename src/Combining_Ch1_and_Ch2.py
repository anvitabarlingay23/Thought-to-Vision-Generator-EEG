import json
import os
import numpy as np
from scipy import interpolate

# --- CONFIGURATION ---
PRE_MS = 500
POST_MS = 1000
SAMPLE_RATE = 250

# --- HARDWARE-SPECIFIC CONVERSION PARAMETERS ---
# VERIFY THESE VALUES MATCH YOUR HARDWARE
# --- For Arduino Uno ---
ADC_RESOLUTION = 1024.0
REFERENCE_VOLTAGE = 5.0

# !! CRITICAL !!
# Updated with the correct gain value based on the BioAmp EXG Pill schematic.
AMPLIFIER_GAIN = 8200.0
# -----------------------------------------------------------------

def parse_raw_data(log_file_path):
    """Reads a raw_serial.log file and separates it into timestamps and ADC values."""
    samples = []
    print(f"Reading and parsing raw data from: {log_file_path}")
    try:
        with open(log_file_path, "r") as f:
            for line in f:
                if line.startswith("S,"):
                    try:
                        _, micros_str, adc_str = line.strip().split(",", 2)
                        samples.append((int(micros_str), int(adc_str)))
                    except (ValueError, IndexError):
                        pass # Ignore any malformed lines
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_path}")
        return None, None

    if not samples:
        print(f"Warning: No valid EEG samples were found in {log_file_path}.")
        return None, None

    print(f"Found {len(samples)} samples.")
    sample_times = np.array([s[0] for s in samples], dtype=np.int64)
    sample_vals = np.array([s[1] for s in samples], dtype=np.float32)
    return sample_times, sample_vals

def extract_epoch(marker_micros, sample_times, sample_vals, out_samples):
    """Extracts and resamples a single epoch window around a given marker timestamp."""
    if marker_micros is None or sample_times is None:
        return None
        
    start_micros = marker_micros - int(PRE_MS * 1000)
    end_micros = marker_micros + int(POST_MS * 1000)
    
    idx = np.where((sample_times >= start_micros) & (sample_times <= end_micros))[0]
    
    if len(idx) < 2:
        return None
        
    times_in_window = sample_times[idx]
    vals_in_window = sample_vals[idx]
    
    new_timeline = np.linspace(start_micros, end_micros, out_samples)
    
    interpolation_func = interpolate.interp1d(times_in_window, vals_in_window, kind='linear', bounds_error=False, fill_value='extrapolate')
    resampled_vals = interpolation_func(new_timeline)
    
    return resampled_vals

def convert_adc_to_microvolts(adc_epoch):
    """
    Converts an array of raw ADC values into microvolts (uV),
    with DC offset removal.
    """
    if adc_epoch is None:
        return None
    
    # --- START: CORRECTED LOGIC ---
    
    # 1. Convert ADC values to voltage
    voltage_epoch = (adc_epoch / ADC_RESOLUTION) * REFERENCE_VOLTAGE
    
    # 2. Remove the DC Offset by subtracting the mean of the signal
    # This re-centers the signal around 0V.
    mean_voltage = np.mean(voltage_epoch)
    ac_voltage_epoch = voltage_epoch - mean_voltage
    
    # 3. Convert the AC voltage to microvolts, accounting for amplifier gain
    microvolts_epoch = (ac_voltage_epoch / AMPLIFIER_GAIN) * 1_000_000
    
    # --- END: CORRECTED LOGIC ---
    
    return microvolts_epoch

def main():
    """Main function to find, process, and combine session data."""
    print("--- Two-Channel EEG Session Combiner (Corrected) ---")
    subject_id = input("Enter the Subject ID to process (e.g., sub01): ")
    
    subject_dir = os.path.join("session_output", subject_id)
    if not os.path.isdir(subject_dir):
        raise SystemExit(f"Error: Subject directory not found at '{subject_dir}'")

    session1_dir = os.path.join(subject_dir, "session_1")
    session2_dir = os.path.join(subject_dir, "session_2")
    
    try:
        with open(os.path.join(session1_dir, "markers.json"), "r") as f:
            markers1 = json.load(f)
        with open(os.path.join(session2_dir, "markers.json"), "r") as f:
            markers2 = json.load(f)
    except FileNotFoundError as e:
        raise SystemExit(f"Error: Could not find marker files. Details: {e}")

    s1_times, s1_vals = parse_raw_data(os.path.join(session1_dir, "raw_serial.log"))
    s2_times, s2_vals = parse_raw_data(os.path.join(session2_dir, "raw_serial.log"))

    if s1_times is None or s2_times is None:
        raise SystemExit("Error: Could not load valid sample data from one or both sessions. Exiting.")

    num_trials_to_process = min(len(markers1), len(markers2))
    if len(markers1) != len(markers2):
        print(f"\n--- WARNING: Trial Count Mismatch ---")
        print(f"Will process the {num_trials_to_process} common trials.\n")
    
    total_ms = PRE_MS + POST_MS
    target_samples = int(SAMPLE_RATE * (total_ms / 1000.0))
    
    final_combined_data = []
    processed_count = 0
    
    print(f"\nProcessing and converting {num_trials_to_process} trials...")

    for i in range(num_trials_to_process):
        m1 = markers1[i]
        m2 = markers2[i]
        
        assert m1["label"] == m2["label"], f"Label mismatch at trial index {i}"
        assert m1["image"] == m2["image"], f"Image name mismatch at trial index {i}"

        raw_epoch_ch1 = extract_epoch(m1['micros'], s1_times, s1_vals, target_samples)
        raw_epoch_ch2 = extract_epoch(m2['micros'], s2_times, s2_vals, target_samples)
        
        if raw_epoch_ch1 is not None and raw_epoch_ch2 is not None:
            microvolts_epoch_ch1 = convert_adc_to_microvolts(raw_epoch_ch1)
            microvolts_epoch_ch2 = convert_adc_to_microvolts(raw_epoch_ch2)

            record = {
                "subject": subject_id,
                "label": m1["label"],
                "image": m1["image"],
                "eeg_data": [
                    microvolts_epoch_ch1.tolist(),
                    microvolts_epoch_ch2.tolist()
                ]
            }
            final_combined_data.append(record)
            processed_count += 1

    output_filename = os.path.join(subject_dir, f"{subject_id}_combined_two_channel_microvolts.json")
    with open(output_filename, "w") as f:
        json.dump(final_combined_data, f, indent=2)
        
    print(f"\n--- Processing Complete ---")
    print(f"Successfully processed and combined {processed_count}/{num_trials_to_process} trials.")
    print(f"Final dataset saved to: {output_filename}")

if __name__ == "__main__":
    main()

