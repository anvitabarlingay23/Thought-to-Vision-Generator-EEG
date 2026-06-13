import os
import json

# Root directory where subject folders exist
ROOT_DIR = "session_output"

def count_trials_in_file(filepath):
    try:
        with open(filepath, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            return len(data)
        else:
            print(f"⚠ Unexpected format in {filepath}")
            return 0

    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}")
        return 0


def main():
    print("=== Counting Trials in Filtered Combined Files ===\n")

    if not os.path.isdir(ROOT_DIR):
        print(f"❌ Folder '{ROOT_DIR}' not found.")
        return

    total_trials = 0
    subject_count = 0

    for subject in os.listdir(ROOT_DIR):
        subject_path = os.path.join(ROOT_DIR, subject)

        if not os.path.isdir(subject_path):
            continue

        # Adjust this filename if your filtered file has a different name
        filtered_file = os.path.join(
            subject_path,
            f"{subject}_filtered_eeg.json"
        )

        if os.path.exists(filtered_file):
            trials = count_trials_in_file(filtered_file)
            print(f"{subject}: {trials} trials")
            total_trials += trials
            subject_count += 1
        else:
            print(f"{subject}: ⚠ filtered file not found")

    print("\n==============================")
    print(f"Subjects processed: {subject_count}")
    print(f"Total filtered trials: {total_trials}")
    print("==============================")

if __name__ == "__main__":
    main()