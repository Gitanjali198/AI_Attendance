"""
Real-time Face Recognition Kiosk & Frame Inference for AI Attendance System.
Powered by Deep SFace (Angle-Invariant MobileFaceNet) + YuNet (5-Point Landmark Face Detector)
with automatic fallback to classical OpenCV LBPH.
Identifies employees from ANY viewing angle (frontal, tilted, turned, up, down) and records Check-In / Check-Out.
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
    HAAR_CASCADE_PATH, TRAINER_MODEL_PATH, LABELS_PATH, EMBEDDINGS_PATH,
    DEFAULT_CONFIDENCE_THRESHOLD, FACE_IMAGE_SIZE, DEFAULT_CAMERA_INDEX,
    SFACE_MATCH_THRESHOLD, YUNET_SCORE_THRESHOLD
)
import backend.config as cfg
MIN_CONFIDENCE_PERCENT = getattr(cfg, "MIN_CONFIDENCE_PERCENT", 58)
from backend.attendance import mark_attendance
from recognition.sface_engine import (
    detect_faces, align_and_extract_feature, load_embeddings,
    match_face_embedding, ensure_models_downloaded
)

def play_chime(frequency=1400, duration=150):
    """Plays a pleasant non-blocking audio beep on Windows."""
    def _beep():
        try:
            import winsound
            winsound.Beep(frequency, duration)
        except Exception:
            pass
    threading.Thread(target=_beep, daemon=True).start()

def load_lbph_engine():
    """
    Fallback: Loads Haar cascade, LBPH recognizer, and labels mapping.
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
    Uses Deep SFace + YuNet as primary high-accuracy, angle-invariant recognition engine.
    Falls back gracefully to Haar Cascade + LBPH if models are uninitialized.
    Returns:
    [
        {
            "box": (x, y, w, h),
            "landmarks": [(x,y), ...],
            "label_id": int,
            "raw_name": str,
            "employee_id": str,
            "employee_name": str,
            "confidence_distance": float,
            "confidence_percent": int,
            "is_recognized": bool,
            "engine": str
        }, ...
    ]
    """
    if frame is None or frame.size == 0:
        return []

    # Check for Deep SFace Embeddings
    embeddings_db = load_embeddings()
    results = []

    # 1. PRIMARY ENGINE: Deep SFace + YuNet Multi-Angle AI
    if embeddings_db:
        try:
            has_faces, detected_faces = detect_faces(frame, score_threshold=YUNET_SCORE_THRESHOLD)
            if has_faces and len(detected_faces) > 0:
                for face_info in detected_faces:
                    raw_face = face_info["raw_face"]
                    box = face_info["box"]
                    landmarks = face_info["landmarks"]

                    query_feat = align_and_extract_feature(frame, raw_face)
                    match_res = match_face_embedding(query_feat, embeddings_db, threshold=SFACE_MATCH_THRESHOLD)

                    results.append({
                        "box": box,
                        "landmarks": landmarks,
                        "label_id": -1,
                        "raw_name": match_res["raw_name"],
                        "employee_id": match_res["employee_id"],
                        "employee_name": match_res["employee_name"],
                        "confidence_distance": round(1.0 - match_res["cosine_similarity"], 3),
                        "cosine_similarity": match_res["cosine_similarity"],
                        "confidence_percent": match_res["confidence_percent"],
                        "is_recognized": match_res["is_recognized"],
                        "engine": "Deep SFace AI (Angle-Invariant)"
                    })

                return results
        except Exception as e:
            print(f"[Recognition Engine] Deep SFace failed, falling back to LBPH: {e}")

    # 2. FALLBACK ENGINE: Classical Haar Cascade + LBPH
    try:
        detector, recognizer, labels = load_lbph_engine()

        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, FACE_IMAGE_SIZE)

            label_id, distance = recognizer.predict(face_resized)

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
                "landmarks": [],
                "label_id": label_id,
                "raw_name": raw_name,
                "employee_id": emp_id,
                "employee_name": emp_name,
                "confidence_distance": round(distance, 1),
                "confidence_percent": confidence_percent,
                "is_recognized": is_recognized,
                "engine": "Classical LBPH"
            })

    except Exception as e:
        print(f"[Recognition Engine Error] {e}")

    return results

def run_face_kiosk(initial_mode="checkin", camera_index=DEFAULT_CAMERA_INDEX, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    """
    Launches an interactive OpenCV live camera kiosk with real-time HUD.
    Features:
    - Deep Multi-Angle SFace AI face tracking & 5 facial landmarks overlay
    - Press 'm' to toggle mode (Check-In <-> Check-Out)
    - Press 'c' to clear cooldown cache
    - Press 'q' or [ESC] to exit
    - Cooldown period to avoid duplicate check-ins
    - Audio chime feedback on successful verification
    """
    embeddings_db = load_embeddings()
    use_deep_ai = len(embeddings_db) > 0

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
    if use_deep_ai:
        print("   ENGINE: Deep SFace Multi-Angle AI (Angle & Pose Invariant)")
    else:
        print("   ENGINE: Classical LBPH Recognizer (Train model for Deep AI)")
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

        # Top HUD Header Bar
        cv2.rectangle(frame, (0, 0), (w_frame, 55), (20, 20, 20), -1)
        mode_text = f"MODE: {mode.upper()}"
        mode_color = (0, 255, 128) if mode == "checkin" else (0, 165, 255)
        engine_tag = "DEEP MULTI-ANGLE AI" if use_deep_ai else "CLASSICAL LBPH"
        cv2.putText(frame, f"AI ATTENDANCE KIOSK [{engine_tag}]", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, mode_text, (w_frame - 230, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

        current_time = time.time()
        predictions = predict_single_frame(frame, confidence_threshold=confidence_threshold)

        for item in predictions:
            x, y, w, h = item["box"]
            is_match = item["is_recognized"]
            emp_name = item["employee_name"]
            raw_name = item["raw_name"]
            conf_pct = item["confidence_percent"]
            landmarks = item.get("landmarks", [])

            # Draw 5 facial landmark dots if available (eyes, nose, mouth corners)
            if landmarks:
                for (lx, ly) in landmarks:
                    cv2.circle(frame, (int(lx), int(ly)), 3, (0, 255, 255), -1)

            if is_match:
                box_color = (0, 255, 0)

                # Check cooldown
                last_time = processed_cooldowns.get(raw_name, 0)
                if current_time - last_time > COOLDOWN_SECONDS:
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
                tag_y = y - 10 if y - 10 > 25 else y + h + 25
                cv2.rectangle(frame, (x, tag_y - 20), (x + 210, tag_y + 4), (20, 20, 20), -1)
                cv2.putText(frame, f"{emp_name} ({conf_pct}%)", (x + 5, tag_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            else:
                box_color = (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
                tag_y = y - 10 if y - 10 > 25 else y + h + 25
                cv2.rectangle(frame, (x, tag_y - 20), (x + 190, tag_y + 4), (20, 20, 20), -1)
                cv2.putText(frame, f"Unknown ({conf_pct}%)", (x + 5, tag_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Bottom HUD Notification Banner
        if current_time - banner_time < 4.0 and last_status_banner:
            cv2.rectangle(frame, (0, h_frame - 45), (w_frame, h_frame), (25, 25, 25), -1)
            cv2.putText(frame, last_status_banner, (25, h_frame - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, banner_color, 2)

        cv2.imshow("AI Attendance Kiosk (Deep Multi-Angle Recognition)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('m'):
            mode = "checkout" if mode == "checkin" else "checkin"
            last_status_banner = f"Switched mode to {mode.upper()}"
            banner_color = (255, 255, 0)
            banner_time = current_time
        elif key == ord('c'):
            processed_cooldowns.clear()
            last_status_banner = "Cooldowns cleared"
            banner_color = (255, 255, 255)
            banner_time = current_time

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI Face Attendance Kiosk")
    parser.add_argument("--mode", choices=["checkin", "checkout"], default="checkin", help="Initial mode")
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX, help="Camera index")
    args = parser.parse_args()

    run_face_kiosk(initial_mode=args.mode, camera_index=args.camera)
