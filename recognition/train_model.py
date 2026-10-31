"""
Face recognition model trainer for AI Attendance System.
Trains an OpenCV LBPH (Local Binary Patterns Histograms) model on employee face datasets.
Can be run as a script or invoked as a function from the web dashboard.
"""
import os
import sys
import json
import time
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.config import (
    DATASET_DIR, TRAINER_DIR, TRAINER_MODEL_PATH,
    LABELS_PATH, FACE_IMAGE_SIZE
)

def train_model():
    """
    Scans the dataset directory, trains the LBPH face recognizer,
    and updates trainer.yml and labels.json.
    Returns a dictionary with training results.
    """
    start_time = time.time()
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    labels = []
    label_map = {}
    current_label = 0

    if not DATASET_DIR.exists():
        return {
            "success": False,
            "message": f"Dataset directory '{DATASET_DIR}' does not exist.",
            "num_employees": 0,
            "total_samples": 0
        }

    employee_folders = [f for f in DATASET_DIR.iterdir() if f.is_dir()]

    if not employee_folders:
        return {
            "success": False,
            "message": "No employee image folders found in employee_images/.",
            "num_employees": 0,
            "total_samples": 0
        }

    total_images_loaded = 0

    for folder in sorted(employee_folders, key=lambda x: x.name):
        folder_name = folder.name
        label_map[current_label] = folder_name
        employee_sample_count = 0

        for img_path in folder.iterdir():
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                continue

            try:
                # Read image in grayscale
                image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue

                # Ensure uniform size
                if image.shape != FACE_IMAGE_SIZE:
                    image = cv2.resize(image, FACE_IMAGE_SIZE)

                faces.append(image)
                labels.append(current_label)
                employee_sample_count += 1
                total_images_loaded += 1
            except Exception as e:
                print(f"Skipping corrupt image {img_path}: {e}")

        if employee_sample_count > 0:
            current_label += 1
        else:
            # Remove empty label from map
            del label_map[current_label]

    if not faces:
        return {
            "success": False,
            "message": "No valid face images found to train the model.",
            "num_employees": 0,
            "total_samples": 0
        }

    labels_array = np.array(labels)

    # Train recognizer
    recognizer.train(faces, labels_array)

    # Ensure trainer directory exists
    TRAINER_DIR.mkdir(parents=True, exist_ok=True)

    # Save model and labels
    recognizer.save(str(TRAINER_MODEL_PATH))
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=4)

    duration = round(time.time() - start_time, 2)

    return {
        "success": True,
        "message": f"Model successfully trained in {duration}s on {len(label_map)} employees with {total_images_loaded} face samples.",
        "num_employees": len(label_map),
        "total_samples": total_images_loaded,
        "labels": label_map,
        "duration_seconds": duration
    }

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("\n[AI Trainer] Scanning face dataset and training LBPH model...")
    result = train_model()
    if result["success"]:
        print(f"[SUCCESS] {result['message']}")
        print("\nEmployee Class Mapping:")
        for k, v in result["labels"].items():
            print(f"  ID {k} -> {v}")
    else:
        print(f"[FAILED] Training failed: {result['message']}")
