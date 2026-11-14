"""
Employee registration and face dataset collection for AI Attendance System.
Captures face samples using webcam or processes uploaded images from any angle,
saves them to employee_images/{id}_{name}/, and updates deep AI embeddings.
"""
import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.config import (
    HAAR_CASCADE_PATH, DATASET_DIR, FACE_IMAGE_SIZE,
    BURST_CAPTURE_COUNT, DEFAULT_CAMERA_INDEX
)
from backend.employee import add_employee
from recognition.sface_engine import detect_faces

def get_face_detector():
    """Fallback: Loads and validates Haar Cascade face detector."""
    if not HAAR_CASCADE_PATH.exists():
        raise FileNotFoundError(f"Haar Cascade XML not found at {HAAR_CASCADE_PATH}")
    detector = cv2.CascadeClassifier(str(HAAR_CASCADE_PATH))
    if detector.empty():
        raise RuntimeError("Failed to load Haar Cascade classifier.")
    return detector

def process_and_save_face_images(emp_id, emp_name, images_data_list):
    """
    Takes a list of image byte buffers or numpy images (from Streamlit file uploader/camera input).
    Uses Deep YuNet detector to find and crop faces at ANY angle (frontal, profile, tilted).
    Saves cropped faces to employee_images/{emp_id}_{emp_name}/.
    Returns (saved_count, message).
    """
    save_folder = DATASET_DIR / f"{emp_id}_{emp_name}"
    save_folder.mkdir(parents=True, exist_ok=True)

    # Count existing images in directory
    existing_count = len(list(save_folder.glob("*.jpg")))
    saved_count = 0

    for idx, item in enumerate(images_data_list):
        try:
            if isinstance(item, np.ndarray):
                frame = item
            elif hasattr(item, "read"):
                file_bytes = np.asarray(bytearray(item.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            elif isinstance(item, (bytes, bytearray)):
                file_bytes = np.asarray(bytearray(item), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            else:
                continue

            if frame is None or frame.size == 0:
                continue

            # 1. Primary: Use Deep YuNet Face Detector (Multi-Angle)
            face_crop = None
            try:
                has_faces, detected = detect_faces(frame, score_threshold=0.50)
                if has_faces and len(detected) > 0:
                    # Pick largest detected face
                    best_face = max(detected, key=lambda f: f["box"][2] * f["box"][3])
                    x, y, w, h = best_face["box"]
                    # Add 10% margin around bounding box for context
                    h_img, w_img = frame.shape[:2]
                    pad_x = int(w * 0.1)
                    pad_y = int(h * 0.1)
                    x1 = max(0, x - pad_x)
                    y1 = max(0, y - pad_y)
                    x2 = min(w_img, x + w + pad_x)
                    y2 = min(h_img, y + h + pad_y)
                    face_crop = frame[y1:y2, x1:x2]
            except Exception as e:
                print(f"[Register Employee] YuNet detection error: {e}")

            # 2. Fallback: Haar Cascade if YuNet didn't find face
            if face_crop is None:
                try:
                    detector = get_face_detector()
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50))
                    if len(faces) > 0:
                        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                        x, y, w, h = faces[0]
                        face_crop = frame[y:y+h, x:x+w]
                except Exception:
                    pass

            # 3. Last fallback: use the frame directly if it's already a square crop
            if face_crop is None:
                if frame.shape[0] >= 50 and frame.shape[1] >= 50:
                    face_crop = frame

            if face_crop is not None:
                face_resized = cv2.resize(face_crop, (200, 200))
                existing_count += 1
                img_path = save_folder / f"{existing_count}.jpg"
                cv2.imwrite(str(img_path), face_resized)
                saved_count += 1

        except Exception as e:
            print(f"Error processing image {idx}: {e}")

    # Auto-retrain model in background so new employee is immediately active
    if saved_count > 0:
        try:
            from recognition.train_model import train_model
            train_model()
        except Exception as e:
            print(f"[Register Employee] Auto-retraining error: {e}")

    return saved_count, f"Saved {saved_count} multi-angle biometric face sample(s) for {emp_name}."

def capture_employee_via_webcam(emp_id, emp_name, target_samples=BURST_CAPTURE_COUNT, camera_index=DEFAULT_CAMERA_INDEX):
    """
    Opens interactive OpenCV camera window to capture face dataset.
    Returns (success: bool, captured_count: int, message: str)
    """
    save_folder = DATASET_DIR / f"{emp_id}_{emp_name}"
    save_folder.mkdir(parents=True, exist_ok=True)

    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        return False, 0, f"Cannot open camera at index {camera_index}."

    count = 0
    print("\n[AI Registration] Look into the camera and slightly tilt/rotate your head.")
    print("Press [ESC] or 'q' to stop early.\n")

    while count < target_samples:
        ret, frame = camera.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h_frame, w_frame = frame.shape[:2]

        has_faces, detected = detect_faces(frame, score_threshold=0.55)

        # HUD Overlay
        cv2.rectangle(frame, (0, 0), (w_frame, 60), (30, 30, 30), -1)
        cv2.putText(
            frame,
            f"Employee: {emp_name} (ID: {emp_id})",
            (20, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        progress_text = f"Samples: {count}/{target_samples} ({(count / target_samples * 100):.0f}%)"
        cv2.putText(
            frame,
            progress_text,
            (20, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 128),
            2
        )

        if has_faces and len(detected) > 0:
            best_face = max(detected, key=lambda f: f["box"][2] * f["box"][3])
            x, y, w, h = best_face["box"]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Capturing Multi-Angle Biometrics...", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

            # Save sample
            count += 1
            face_crop = frame[y:y+h, x:x+w]
            if face_crop.size > 0:
                face_resized = cv2.resize(face_crop, (200, 200))
                img_path = save_folder / f"{count}.jpg"
                cv2.imwrite(str(img_path), face_resized)

        else:
            cv2.putText(frame, "Align your face in view...", (w_frame // 2 - 120, h_frame // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)

        cv2.imshow("Register Employee - Biometric Multi-Angle Capture", frame)

        key = cv2.waitKey(80) & 0xFF
        if key == ord('q') or key == 27:
            break

    camera.release()
    cv2.destroyAllWindows()

    if count > 0:
        # Trigger model retraining
        from recognition.train_model import train_model
        train_model()
        return True, count, f"Successfully captured {count} multi-angle biometric samples for {emp_name}."
    else:
        return False, 0, "No face samples were captured."

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Employee Biometric Registration")
    parser.add_argument("--id", type=str, help="Employee ID")
    parser.add_argument("--name", type=str, help="Employee Name")
    parser.add_argument("--dept", type=str, help="Department")
    parser.add_argument("--email", type=str, default="", help="Email")
    args = parser.parse_args()

    if args.id and args.name and args.dept:
        success, msg = add_employee(args.id, args.name, args.dept, args.email)
        print(f"[DB Status] {msg}")
        if success:
            cap_success, count, cap_msg = capture_employee_via_webcam(args.id, args.name)
            print(f"[Capture Status] {cap_msg}")
    else:
        print("\n=== AI Face Attendance - Employee Registration CLI ===")
        emp_id = input("Enter Employee ID (e.g., 104): ").strip()
        emp_name = input("Enter Full Name: ").strip()
        dept = input("Enter Department: ").strip()
        email = input("Enter Email (optional): ").strip()

        if emp_id and emp_name and dept:
            success, msg = add_employee(emp_id, emp_name, dept, email)
            print(f"[DB Status] {msg}")
            if success:
                cap_success, count, cap_msg = capture_employee_via_webcam(emp_id, emp_name)
                print(f"[Capture Status] {cap_msg}")
        else:
            print("[Error] ID, Name, and Department are required.")
