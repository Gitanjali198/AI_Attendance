"""
Employee registration and face dataset collection for AI Attendance System.
Captures face samples using webcam or processes uploaded images, saves them to employee_images/,
and registers employee details in the database.
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

def get_face_detector():
    """Loads and validates Haar Cascade face detector."""
    if not HAAR_CASCADE_PATH.exists():
        raise FileNotFoundError(f"Haar Cascade XML not found at {HAAR_CASCADE_PATH}")
    detector = cv2.CascadeClassifier(str(HAAR_CASCADE_PATH))
    if detector.empty():
        raise RuntimeError("Failed to load Haar Cascade classifier.")
    return detector

def process_and_save_face_images(emp_id, emp_name, images_data_list):
    """
    Takes a list of image byte buffers or numpy images (e.g. from Streamlit file uploader/camera input),
    detects faces, crops, normalizes, and saves them to employee_images/{emp_id}_{emp_name}/.
    Returns (saved_count, message).
    """
    detector = get_face_detector()
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

            if frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))

            # If no face detected with strict params, try slightly relaxed params
            if len(faces) == 0:
                faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))

            if len(faces) > 0:
                # Pick the largest face in frame
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                (x, y, w, h) = faces[0]
                face_crop = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_crop, FACE_IMAGE_SIZE)

                existing_count += 1
                img_path = save_folder / f"{existing_count}.jpg"
                cv2.imwrite(str(img_path), face_resized)
                saved_count += 1
            else:
                # If entire image is already a cropped face, resize and save
                if frame.shape[0] >= 50 and frame.shape[1] >= 50:
                    gray_resized = cv2.resize(gray, FACE_IMAGE_SIZE)
                    existing_count += 1
                    img_path = save_folder / f"{existing_count}.jpg"
                    cv2.imwrite(str(img_path), gray_resized)
                    saved_count += 1
        except Exception as e:
            print(f"Error processing image {idx}: {e}")

    return saved_count, f"Saved {saved_count} face sample(s) for {emp_name}."

def capture_employee_via_webcam(emp_id, emp_name, target_samples=BURST_CAPTURE_COUNT, camera_index=DEFAULT_CAMERA_INDEX):
    """
    Opens interactive OpenCV camera window to capture face dataset.
    Returns (success: bool, captured_count: int, message: str)
    """
    try:
        detector = get_face_detector()
    except Exception as e:
        return False, 0, str(e)

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

        # Flip horizontally for natural mirror feel
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))

        # HUD Overlay
        h_frame, w_frame = frame.shape[:2]
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
            (0, 255, 255),
            2
        )

        for (x, y, w, h) in faces:
            count += 1
            face_crop = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_crop, FACE_IMAGE_SIZE)

            cv2.imwrite(str(save_folder / f"{count}.jpg"), face_resized)

            # Draw green bounding box with corner highlights
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Face #{count}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            break  # one face per frame for quality

        cv2.imshow("Employee Face Registration - Look at Camera", frame)
        key = cv2.waitKey(80) & 0xFF
        if key in [27, ord('q')]:
            break

    camera.release()
    cv2.destroyAllWindows()

    return True, count, f"Registration complete! Captured {count} images."

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("\n==============================================")
    print("   AI Attendance System - Employee Registration")
    print("==============================================\n")

    emp_id = input("Enter Employee ID (e.g. 104): ").strip()
    if not emp_id:
        print("[Error] Employee ID cannot be empty.")
        sys.exit(1)

    emp_name = input("Enter Employee Name (e.g. Alice): ").strip()
    if not emp_name:
        print("[Error] Employee Name cannot be empty.")
        sys.exit(1)

    department = input("Enter Department (e.g. Engineering): ").strip() or "General"
    email = input("Enter Email (optional): ").strip()

    # Save to database
    success, msg = add_employee(emp_id, emp_name, department, email)
    if not success and "already exists" not in msg:
        print(f"[Error] {msg}")
        sys.exit(1)
    print(f"[Database] {msg}")

    print("\nLaunching webcam to capture 50 face samples...")
    success, captured, capture_msg = capture_employee_via_webcam(emp_id, emp_name, target_samples=50)
    print(f"[Capture] {capture_msg}")

    if captured > 0:
        train_now = input("\nWould you like to train the AI model now? (y/n): ").strip().lower()
        if train_now in ["y", "yes"]:
            from recognition.train_model import train_model
            res = train_model()
            if res["success"]:
                print(f"[Model Trained] {res['message']}")
            else:
                print(f"[Training Error] {res['message']}")
