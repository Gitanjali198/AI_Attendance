"""
Face recognition model trainer for AI Attendance System.
Upgraded to train both:
1. Deep Learning Multi-Angle SFace Embeddings (Angle, Pose, and Lighting Invariant)
2. Classical OpenCV LBPH Face Recognizer (Fallback Engine)
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
    LABELS_PATH, EMBEDDINGS_PATH, FACE_IMAGE_SIZE
)
from recognition.sface_engine import (
    detect_faces, align_and_extract_feature, generate_multi_angle_features,
    save_embeddings, ensure_models_downloaded
)

def train_model():
    """
    Scans the dataset directory, trains both the Deep SFace feature embeddings
    (with multi-angle synthetic augmentations) and the LBPH face recognizer,
    and updates face_embeddings.json, trainer.yml, and labels.json.
    Returns a dictionary with training results.
    """
    start_time = time.time()
    TRAINER_DIR.mkdir(parents=True, exist_ok=True)

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

    # Data structures for LBPH
    lbph_faces = []
    lbph_labels = []
    label_map = {}
    current_label = 0

    # Data structures for Deep SFace
    embeddings_db = {}
    total_images_loaded = 0
    total_deep_embeddings = 0

    # Ensure Deep Learning models exist
    has_deep_models = ensure_models_downloaded()

    for folder in sorted(employee_folders, key=lambda x: x.name):
        folder_name = folder.name
        if "_" in folder_name:
            emp_id, emp_name = folder_name.split("_", 1)
        else:
            emp_id, emp_name = folder_name, folder_name

        label_map[current_label] = folder_name
        employee_sample_count = 0
        emp_features = []

        for img_path in folder.iterdir():
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                continue

            try:
                # Read color image for Deep Learning
                bgr_img = cv2.imread(str(img_path))
                if bgr_img is None:
                    continue

                total_images_loaded += 1
                employee_sample_count += 1

                # 1. Deep SFace Feature Extraction & Multi-Angle Augmentation
                if has_deep_models:
                    has_face, detected_faces = detect_faces(bgr_img, score_threshold=0.50)
                    if has_face and len(detected_faces) > 0:
                        raw_face = detected_faces[0]["raw_face"]
                        # Generate multi-angle features (canonical + horizontal flip + tilted + yaw)
                        multi_feats = generate_multi_angle_features(bgr_img, raw_face)
                        emp_features.extend(multi_feats)

                # 2. LBPH Preparation (Grayscale resized)
                gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
                if gray_img.shape != FACE_IMAGE_SIZE:
                    gray_img = cv2.resize(gray_img, FACE_IMAGE_SIZE)

                lbph_faces.append(gray_img)
                lbph_labels.append(current_label)

            except Exception as e:
                print(f"[AI Trainer] Skipping image {img_path}: {e}")

        # If we collected deep features, store in embeddings_db
        if emp_features:
            mean_feature = np.mean(emp_features, axis=0)
            norm = np.linalg.norm(mean_feature)
            if norm > 0:
                mean_feature = mean_feature / norm

            embeddings_db[folder_name] = {
                "employee_id": emp_id,
                "employee_name": emp_name,
                "features": emp_features,
                "mean_feature": mean_feature,
                "num_embeddings": len(emp_features)
            }
            total_deep_embeddings += len(emp_features)

        if employee_sample_count > 0:
            current_label += 1
        else:
            if current_label in label_map:
                del label_map[current_label]

    if not lbph_faces and not embeddings_db:
        return {
            "success": False,
            "message": "No valid face images found to train the model.",
            "num_employees": 0,
            "total_samples": 0
        }

    # Save Deep SFace Embeddings Database
    deep_saved = False
    if embeddings_db:
        deep_saved = save_embeddings(embeddings_db)

    # Train and Save Classical LBPH Recognizer (Fallback)
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(lbph_faces, np.array(lbph_labels))
        recognizer.save(str(TRAINER_MODEL_PATH))
        with open(LABELS_PATH, "w", encoding="utf-8") as f:
            json.dump(label_map, f, indent=4)
        lbph_trained = True
    except Exception as e:
        print(f"[AI Trainer Warning] LBPH training encountered: {e}")
        lbph_trained = False

    duration = round(time.time() - start_time, 2)

    msg = (
        f"AI Model successfully trained in {duration}s on {len(label_map)} employees "
        f"({total_images_loaded} images $\\rightarrow$ {total_deep_embeddings} multi-angle deep embeddings generated)."
    )

    return {
        "success": True,
        "message": msg,
        "num_employees": len(label_map),
        "total_samples": total_images_loaded,
        "total_deep_embeddings": total_deep_embeddings,
        "labels": label_map,
        "deep_embeddings_active": deep_saved,
        "lbph_active": lbph_trained,
        "duration_seconds": duration
    }

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("\n[AI Trainer] Training Deep Multi-Angle SFace AI & LBPH models...")
    result = train_model()
    if result["success"]:
        print(f"[SUCCESS] {result['message']}")
        print("\nEmployee Class Mapping:")
        for k, v in result["labels"].items():
            print(f"  ID {k} -> {v}")
    else:
        print(f"[FAILED] Training failed: {result['message']}")
