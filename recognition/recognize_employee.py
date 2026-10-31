"""
Real-time Face Recognition Kiosk for AI Attendance System.
Uses OpenCV LBPH face recognizer to identify employees and record Check-In / Check-Out attendance.
Includes interactive HUD, keyboard mode toggling ('m'), and single-frame recognition API for Streamlit.
"""
import os
import sys
import json
import time
import threading
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.config import (
    HAAR_CASCADE_PATH, TRAINER_MODEL_PATH, LABELS_PATH,
    DEFAULT_CONFIDENCE_THRESHOLD, FACE_IMAGE_SIZE, DEFAULT_CAMERA_INDEX
)
import backend.config as cfg
MIN_CONFIDENCE_PERCENT = getattr(cfg, "MIN_CONFIDENCE_PERCENT", 58)
from backend.attendance import mark_attendance

def play_chime(frequency=1400, duration=150):
    """Plays a pleasant non-blocking audio beep on Windows."""
    def _beep():
        try:
            import winsound
            winsound.Beep(frequency, duration)
        except Exception:
            pass
    threading.Thread(target=_beep, daemon=True).start()

def load_recognition_engine():
    """
    Loads Haar cascade, LBPH recognizer, and labels mapping.
    Returns (detector, recognizer, labels) or raises error.
    """
    if not HAAR_CASCADE_PATH.exists():
        raise FileNotFoundError(f"Haar Cascade not found at {HAAR_CASCADE_PATH}")

    detector = cv2.CascadeClassifier(str(HAAR_CASCADE_PATH))
    if detector.empty():
        raise RuntimeError("Failed to load Haar Cascade classifier.")

    if not TRAINER_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model '{TRAINER_MODEL_PATH}' not found. Please train the model first."
        )

    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Labels mapping '{LABELS_PATH}' not found. Please train the model first."
        )

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(TRAINER_MODEL_PATH))

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

    return detector, recognizer, labels

def predict_single_frame(frame, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD, min_confidence_percent=MIN_CONFIDENCE_PERCENT, **kwargs):
    """
    Processes a single frame (numpy array RGB or BGR).
    Detects faces, runs prediction, and returns:
    [
        {
            "box": (x, y, w, h),
            "label_id": int,
            "raw_name": str,
            "employee_id": str,
            "employee_name": str,
            "confidence_distance": float,
            "confidence_percent": int,
            "is_recognized": bool
        }, ...
    ]
    """
    # Fallback if passed through kwargs or positional
    if "min_confidence_percent" in kwargs:
        min_confidence_percent = kwargs["min_confidence_percent"]
    elif "min_confidence" in kwargs:
        min_confidence_percent = kwargs["min_confidence"]

    detector, recognizer, labels = load_recognition_engine()

    if len(frame.shape) == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
    results = []

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, FACE_IMAGE_SIZE)

        label_id, distance = recognizer.predict(face_resized)

        # Distance <= 45 and confidence >= 58% denotes a genuine match (0 is perfect)
        confidence_percent = max(0, min(100, round(100 * (1 - distance / 100))))
        is_recognized = (
            distance <= confidence_threshold
            and confidence_percent >= min_confidence_percent
            and str(label_id) in labels
        )

        raw_name = labels.get(str(label_id), "Unknown") if is_recognized else "Unknown"

        if is_recognized and "_" in raw_name:
            emp_id, emp_name = raw_name.split("_", 1)
        elif is_recognized:
            emp_id, emp_name = raw_name, raw_name
        else:
            emp_id, emp_name = "N/A", "Unknown"

        results.append({
            "box": (x, y, w, h),
            "label_id": label_id,
            "raw_name": raw_name,
            "employee_id": emp_id,
            "employee_name": emp_name,
            "confidence_distance": round(distance, 1),
            "confidence_percent": confidence_percent,
            "is_recognized": is_recognized
        })

    return results

def run_face_kiosk(initial_mode="checkin", camera_index=DEFAULT_CAMERA_INDEX, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    """
    Launches an interactive OpenCV live camera kiosk with real-time HUD.
    Features:
    - Press 'm' to toggle mode (Check-In <-> Check-Out)
    - Press 'q' or [ESC] to exit
    - Cooldown period to avoid spamming database
    - Visual alerts and sound feedback
    """
    try:
        detector, recognizer, labels = load_recognition_engine()
    except Exception as e:
        print(f"[Error] {e}")
        return

    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        print(f"[Error] Unable to access camera at index {camera_index}.")
        return

    mode = initial_mode
    processed_cooldowns = {}   # emp_key -> last_processed_timestamp
    COOLDOWN_SECONDS = 6.0

    last_status_banner = ""
    banner_color = (0, 255, 0)
    banner_time = 0

    print("\n=======================================================")
    print("   AI Attendance Live Kiosk - Camera Running")
    print("=======================================================")
    print("   Controls:")
    print("   [m] : Switch Mode (Check-In <-> Check-Out)")
    print("   [c] : Clear Cooldowns")
    print("   [q] or [ESC] : Exit Kiosk")
    print("=======================================================\n")

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h_frame, w_frame = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))

        # Top HUD Bar
        cv2.rectangle(frame, (0, 0), (w_frame, 55), (20, 20, 20), -1)
        mode_text = f"MODE: {mode.upper()}"
        mode_color = (0, 255, 128) if mode == "checkin" else (0, 165, 255)
        cv2.putText(frame, "AI ATTENDANCE KIOSK", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, mode_text, (w_frame - 240, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

        current_time = time.time()

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, FACE_IMAGE_SIZE)

            label_id, distance = recognizer.predict(face_resized)
            confidence_percent = max(0, min(100, round(100 * (1 - distance / 100))))
            is_match = (
                distance <= confidence_threshold
                and confidence_percent >= MIN_CONFIDENCE_PERCENT
                and str(label_id) in labels
            )

            if is_match:
                raw_name = labels[str(label_id)]
                box_color = (0, 255, 0)

                # Check cooldown
                last_time = processed_cooldowns.get(raw_name, 0)
                if current_time - last_time > COOLDOWN_SECONDS:
                    # Mark attendance in database
                    result = mark_attendance(raw_name, mode=mode)
                    processed_cooldowns[raw_name] = current_time

                    status = result.get("status")
                    if status == "checkin_success":
                        att_stat = result.get("attendance_status", "Present")
                        last_status_banner = f"CHECK-IN [{att_stat.upper()}]: {result['employee_name']} at {result['time']} (Rs 100/hr)"
                        banner_color = (0, 255, 0) if att_stat == "Present" else (0, 165, 255)
                        play_chime(1500, 150)
                    elif status == "already_checked_in":
                        last_status_banner = f"ALREADY IN: {result['employee_name']} ({result['time']})"
                        banner_color = (0, 215, 255)
                    elif status == "checkout_success":
                        wage = result.get("salary_earned", 0.0)
                        last_status_banner = f"CHECK-OUT: {result['employee_name']} ({result.get('duration', '')} - Rs {wage:.2f})"
                        banner_color = (0, 200, 255)
                        play_chime(1200, 200)
                    elif status == "already_checked_out":
                        last_status_banner = f"ALREADY OUT: {result['employee_name']}"
                        banner_color = (0, 165, 255)
                    elif status == "not_checked_in":
                        last_status_banner = f"NOT CHECKED IN: {result['employee_name']}"
                        banner_color = (0, 0, 255)

                    banner_time = current_time
                    print(f"[{mode.upper()}] {last_status_banner}")

                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
                # Header label tag
                tag_y = y - 10 if y - 10 > 25 else y + h + 25
                cv2.rectangle(frame, (x, tag_y - 20), (x + 180, tag_y + 4), (20, 20, 20), -1)
                display_name = raw_name.split("_", 1)[1] if "_" in raw_name else raw_name
                cv2.putText(frame, f"{display_name} ({confidence_percent}%)", (x + 5, tag_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            else:
                # Unknown Face
                box_color = (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
                cv2.putText(frame, "Unknown Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        # Bottom status banner (if within 5 seconds of an event)
        if last_status_banner and (current_time - banner_time < 5.0):
            cv2.rectangle(frame, (0, h_frame - 45), (w_frame, h_frame), (15, 15, 15), -1)
            cv2.putText(frame, last_status_banner, (20, h_frame - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, banner_color, 2)
        else:
            # Bottom help instructions
            cv2.rectangle(frame, (0, h_frame - 30), (w_frame, h_frame), (30, 30, 30), -1)
            cv2.putText(frame, "Press 'm' to switch mode | 'q' or ESC to exit", (20, h_frame - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("AI Attendance System - Face Kiosk", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in [27, ord('q'), ord('Q')]:
            break
        elif key in [ord('m'), ord('M')]:
            mode = "checkout" if mode == "checkin" else "checkin"
            last_status_banner = f"Switched mode to {mode.upper()}"
            banner_color = (255, 255, 0)
            banner_time = time.time()
            play_chime(1000, 80)
        elif key in [ord('c'), ord('C')]:
            processed_cooldowns.clear()
            last_status_banner = "Cooldown cache cleared."
            banner_color = (255, 255, 255)
            banner_time = time.time()

    camera.release()
    cv2.destroyAllWindows()
    print("\n[AI Kiosk] Stopped.")

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("\n==========================================")
    print("   AI Attendance System - Live Kiosk")
    print("==========================================")
    print("1. Check In Mode")
    print("2. Check Out Mode")
    choice = input("\nSelect Initial Mode (1/2, default 1): ").strip()
    initial_mode = "checkout" if choice == "2" else "checkin"
    run_face_kiosk(initial_mode=initial_mode)
