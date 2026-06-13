import serial, threading, time, json, os, random
import pygame
import numpy as np
from collections import Counter

# ---------- GET SUBJECT INFO ----------
subject_id = input("Please enter the Subject ID (e.g., sub01): ").strip()
session_num = input("Is this Session 1 or 2? (Enter 1 or 2): ").strip()
if session_num not in ["1", "2"]:
    raise SystemExit("Invalid session number.")
com_port = input("Enter Arduino COM port (e.g., COM4): ").strip()

# ---------- CONFIG ----------
BAUD_RATE = 115200
IMAGE_DIR = "shapes_for_project"
SUBJECT_DIR = os.path.join("session_output", subject_id)
OUTPUT_DIR = os.path.join(SUBJECT_DIR, f"session_{session_num}")
TRIAL_SEED = 12345
FIXATION_MS = 500
IMG_MS = 500
ITI_MS = 1200
SERIAL_TIMEOUT = 5
EXPECTED_IMAGES_PER_LABEL = 50

# ---------- SETUP ----------
os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(TRIAL_SEED)

# ---------- LOAD IMAGES ----------
labels = ["circle", "square", "triangle"]
by_label = {}
print("Loading images...")

for label in labels:
    folder = os.path.join(IMAGE_DIR, label)
    if not os.path.isdir(folder):
        raise SystemExit(f"Missing folder: {folder}")

    images = sorted([f for f in os.listdir(folder)
                     if f.lower().endswith((".png", ".jpg", ".jpeg"))])[:EXPECTED_IMAGES_PER_LABEL]

    if not images:
        raise SystemExit(f"No images found in: {folder}")

    by_label[label] = images

print("✅ Images loaded.")

TOTAL_EXPECTED_MARKERS = len(labels) * EXPECTED_IMAGES_PER_LABEL

# ---------- CONNECT TO ARDUINO ----------
print(f"Connecting to {com_port}...")
ser = serial.Serial(com_port, BAUD_RATE, timeout=1)
time.sleep(2)
print("✅ Arduino connected.")

# ---------- SERIAL THREAD ----------
marker_lock = threading.Lock()
markers = []
last_s_time = time.time()
stop_thread = False

SERIAL_LOG_PATH = os.path.join(OUTPUT_DIR, "raw_serial.log")


def serial_reader():
    global last_s_time
    with open(SERIAL_LOG_PATH, "w") as logfile:
        while not stop_thread:
            try:
                line = ser.readline().decode("utf-8", errors="ignore").strip()

                if not line:
                    time.sleep(0.002)
                    continue

                logfile.write(line + "\n")
                logfile.flush()
                last_s_time = time.time()

                if line.startswith("MARK,"):
                    parts = line.split(",", 3)
                    if len(parts) == 4:
                        _, ms, label, img = parts
                        try:
                            micros = int(ms)
                        except:
                            micros = None

                        with marker_lock:
                            markers.append({
                                "micros": micros,
                                "label": label,
                                "image": img.strip(),
                                "recv_time": time.time()
                            })

            except Exception as e:
                print("Serial thread crash:", e)
                break


threading.Thread(target=serial_reader, daemon=True).start()

# ---------- PYGAME ----------
pygame.init()
win = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
screen_width, screen_height = win.get_size()
pygame.display.set_caption(f"EEG Exp - {subject_id}")
font = pygame.font.SysFont(None, 60)
fixation_font = pygame.font.SysFont(None, 250)


def show_message(text, is_error=False, duration_s=None):
    win.fill((90, 90, 90))
    color = (255, 0, 0) if is_error else (255, 255, 255)
    for i, line in enumerate(text.split("\n")):
        surf = font.render(line, True, color)
        win.blit(surf, surf.get_rect(center=(screen_width // 2, screen_height // 2 + i * 70)))
    pygame.display.flip()
    if duration_s:
        time.sleep(duration_s)


def wait_for_space():
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_SPACE:
                return
        time.sleep(0.05)


def get_background_color(surface):
    arr = pygame.surfarray.array3d(surface)
    t = max(1, int(min(arr.shape[:2]) * 0.1))
    zones = [arr[:, :t], arr[:, -t:], arr[:t], arr[-t:]]
    pixels = np.concatenate([z.reshape(-1, 3) for z in zones])
    return Counter(map(tuple, pixels)).most_common(1)[0][0]


def show_fixation_cross(ms):
    win.fill((0, 0, 0))
    surf = fixation_font.render("+", True, (255, 255, 255))
    win.blit(surf, surf.get_rect(center=(screen_width // 2, screen_height // 2)))
    pygame.display.flip()
    pygame.time.delay(ms)


def show_image(label, fname):
    img = pygame.image.load(os.path.join(IMAGE_DIR, label, fname)).convert()
    scale = min(screen_width * 0.8 / img.get_width(), screen_height * 0.8 / img.get_height())
    img = pygame.transform.smoothscale(img,
          (int(img.get_width() * scale), int(img.get_height() * scale)))

    win.fill(get_background_color(img))
    win.blit(img, img.get_rect(center=(screen_width // 2, screen_height // 2)))
    pygame.display.flip()

    ser.write(f"M,{label},{fname}\n".encode())
    ser.flush()

    pygame.time.delay(IMG_MS)


# ---------- EXPERIMENT ----------
show_message("Press SPACE to begin")
wait_for_space()

try:
    for label in labels:
        show_fixation_cross(FIXATION_MS)

        for fname in by_label[label]:

            # Watchdog
            if time.time() - last_s_time > SERIAL_TIMEOUT:
                show_message("FATAL ERROR\nArduino not responding", True, 6)
                raise SystemExit("Arduino silent")

            show_image(label, fname)

            timeout = time.time() + 2
            while time.time() < timeout:
                with marker_lock:
                    if markers and markers[-1]["image"] == fname:
                        break
                time.sleep(0.01)
            else:
                show_message("FATAL ERROR\nEEG marker missing", True, 6)
                raise SystemExit("Marker timeout")

            win.fill((80, 80, 80))
            pygame.display.flip()
            pygame.time.delay(max(500, int(random.gauss(ITI_MS, 100))))

finally:
    stop_thread = True
    time.sleep(0.3)

    try:
        ser.close()
    except:
        pass

    pygame.quit()

    MARKER_FILE = os.path.join(OUTPUT_DIR, "markers.json")
    with open(MARKER_FILE, "w") as f:
        json.dump(markers, f, indent=2)

    print("\n========== FINAL VALIDATION ==========")

    # ⚠ raw serial log
    if not os.path.exists(SERIAL_LOG_PATH) or os.path.getsize(SERIAL_LOG_PATH) == 0:
        raise SystemExit("FAILED: raw_serial.log empty")

    # ⚠ marker file
    if len(markers) == 0:
        raise SystemExit("FAILED: markers.json empty")

    # ⚠ marker count
    if len(markers) != TOTAL_EXPECTED_MARKERS:
        raise SystemExit(f"FAILED: expected {TOTAL_EXPECTED_MARKERS}, got {len(markers)}")

    # ⚠ matching
    trial_images = [f for L in labels for f in by_label[L]]
    invalid = [m for m in markers if m["image"] not in trial_images]

    if invalid:
        raise SystemExit("FAILED: invalid markers detected")

    print("✅ ALL FILES VALID ✅")
