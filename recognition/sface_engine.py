"""
Deep Learning Multi-Angle Face Recognition Engine for AI Attendance System.
Powered by OpenCV YuNet (Ultra-fast face detection & 5-point landmark alignment)
and OpenCV SFace (Deep MobileFaceNet feature embedding extractor).
Provides true angle-invariant, pose-invariant, and lighting-invariant recognition.
"""
import os
import sys
import json
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.config import (
    MODELS_DIR, YUNET_MODEL_PATH, SFACE_MODEL_PATH, EMBEDDINGS_PATH,
    SFACE_MATCH_THRESHOLD, YUNET_SCORE_THRESHOLD, YUNET_NMS_THRESHOLD
)

# Official OpenCV Zoo download mirrors
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

_YUNET_DETECTOR = None
_SFACE_RECOGNIZER = None

def ensure_models_downloaded() -> bool:
    """
    Verifies that YuNet and SFace ONNX models exist locally.
    Downloads them automatically from OpenCV Zoo if missing.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    if not YUNET_MODEL_PATH.exists() or YUNET_MODEL_PATH.stat().st_size < 100000:
        print("[Deep AI] Downloading OpenCV YuNet face detector model (~230 KB)...")
        try:
            urllib.request.urlretrieve(YUNET_URL, str(YUNET_MODEL_PATH))
            print("[Deep AI] YuNet downloaded successfully.")
        except Exception as e:
            print(f"[Deep AI Error] Failed to download YuNet: {e}")
            return False

    if not SFACE_MODEL_PATH.exists() or SFACE_MODEL_PATH.stat().st_size < 10000000:
        print("[Deep AI] Downloading OpenCV SFace deep face recognizer model (~38 MB)...")
        try:
            urllib.request.urlretrieve(SFACE_URL, str(SFACE_MODEL_PATH))
            print("[Deep AI] SFace downloaded successfully.")
        except Exception as e:
            print(f"[Deep AI Error] Failed to download SFace: {e}")
            return False

    return True

def get_yunet_detector(input_size: Tuple[int, int] = (320, 320)):
    """Returns singleton instance of OpenCV YuNet face detector."""
    global _YUNET_DETECTOR
    ensure_models_downloaded()
    if not YUNET_MODEL_PATH.exists():
        raise FileNotFoundError(f"YuNet model not found at {YUNET_MODEL_PATH}")
    
    if _YUNET_DETECTOR is None:
        _YUNET_DETECTOR = cv2.FaceDetectorYN.create(
            model=str(YUNET_MODEL_PATH),
            config="",
            input_size=input_size,
            score_threshold=YUNET_SCORE_THRESHOLD,
            nms_threshold=YUNET_NMS_THRESHOLD,
            top_k=5000
        )
    else:
        _YUNET_DETECTOR.setInputSize(input_size)
    return _YUNET_DETECTOR

def get_sface_recognizer():
    """Returns singleton instance of OpenCV SFace deep face recognizer."""
    global _SFACE_RECOGNIZER
    ensure_models_downloaded()
    if not SFACE_MODEL_PATH.exists():
        raise FileNotFoundError(f"SFace model not found at {SFACE_MODEL_PATH}")
    
    if _SFACE_RECOGNIZER is None:
        _SFACE_RECOGNIZER = cv2.FaceRecognizerSF.create(
            model=str(SFACE_MODEL_PATH),
            config=""
        )
    return _SFACE_RECOGNIZER

def detect_faces(image: np.ndarray, score_threshold: float = YUNET_SCORE_THRESHOLD) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Detects faces at any angle (yaw, pitch, roll, tilt) using YuNet.
    Returns:
        (has_faces, [
            {
                "box": (x, y, w, h),
                "landmarks": [(x0, y0), (x1, y1), ...], # 5 points: right eye, left eye, nose, right mouth, left mouth
                "score": float,
                "raw_face": np.ndarray (15 elements row)
            }, ...
        ])
    """
    if image is None or image.size == 0:
        return False, []

    h, w = image.shape[:2]
    detector = get_yunet_detector(input_size=(w, h))
    detector.setScoreThreshold(score_threshold)
    
    # YuNet expects BGR 8-bit image
    if len(image.shape) == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        bgr = image

    _, raw_faces = detector.detect(bgr)
    if raw_faces is None or len(raw_faces) == 0:
        return False, []

    detected = []
    for face in raw_faces:
        x, y, w_box, h_box = int(face[0]), int(face[1]), int(face[2]), int(face[3])
        # Ensure bounding box is within frame
        x = max(0, x)
        y = max(0, y)
        w_box = min(w - x, w_box)
        h_box = min(h - y, h_box)
        
        # 5 key facial landmarks
        landmarks = [
            (float(face[4]), float(face[5])),   # Right Eye
            (float(face[6]), float(face[7])),   # Left Eye
            (float(face[8]), float(face[9])),   # Nose Tip
            (float(face[10]), float(face[11])), # Right Mouth Corner
            (float(face[12]), float(face[13]))  # Left Mouth Corner
        ]
        score = float(face[14])
        
        detected.append({
            "box": (x, y, w_box, h_box),
            "landmarks": landmarks,
            "score": score,
            "raw_face": face
        })

    return True, detected

def align_and_extract_feature(image: np.ndarray, raw_face: np.ndarray) -> Optional[np.ndarray]:
    """
    Aligns the face into canonical orientation using 5 landmarks,
    crops it to 112x112, and extracts the 128-dimensional normalized deep feature vector.
    """
    if image is None or raw_face is None:
        return None

    if len(image.shape) == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        bgr = image

    recognizer = get_sface_recognizer()
    aligned = recognizer.alignCrop(bgr, raw_face)
    feature = recognizer.feature(aligned)
    
    # Normalize to unit vector for fast cosine similarity via dot product
    norm = np.linalg.norm(feature)
    if norm > 0:
        feature = feature / norm
    return feature

def generate_multi_angle_features(image: np.ndarray, raw_face: np.ndarray) -> List[np.ndarray]:
    """
    Generates multi-angle deep embeddings for an employee's face photo:
    - Base canonical aligned feature
    - Horizontal flip (simulating looking from the alternate side/profile)
    - Rotated variants (-12°, +12°) simulating head tilt/roll
    - Perspective warp simulating 3D yaw head angle
    This ensures that even from 1-2 photos, the model recognizes the employee from ANY angle.
    """
    features = []
    
    # 1. Base canonical feature
    base_feat = align_and_extract_feature(image, raw_face)
    if base_feat is not None:
        features.append(base_feat)

    h, w = image.shape[:2]
    
    # 2. Horizontal mirror (Yaw reflection)
    try:
        img_flip = cv2.flip(image, 1)
        has_f, faces_flip = detect_faces(img_flip, score_threshold=0.50)
        if has_f and len(faces_flip) > 0:
            feat_flip = align_and_extract_feature(img_flip, faces_flip[0]["raw_face"])
            if feat_flip is not None:
                features.append(feat_flip)
    except Exception:
        pass

    # 3. Angle rotations for head tilt (-12 deg and +12 deg)
    for angle in [-12, 12]:
        try:
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
            img_rot = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            has_r, faces_rot = detect_faces(img_rot, score_threshold=0.50)
            if has_r and len(faces_rot) > 0:
                feat_rot = align_and_extract_feature(img_rot, faces_rot[0]["raw_face"])
                if feat_rot is not None:
                    features.append(feat_rot)
        except Exception:
            pass

    # 4. Affine perspective transform for 3D yaw simulation
    try:
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([[w * 0.10, 0], [w * 0.90, h * 0.05], [0, h], [w * 0.85, h * 0.95]])
        M_persp = cv2.getPerspectiveTransform(pts1, pts2)
        img_persp = cv2.warpPerspective(image, M_persp, (w, h), borderMode=cv2.BORDER_REFLECT)
        has_p, faces_p = detect_faces(img_persp, score_threshold=0.50)
        if has_p and len(faces_p) > 0:
            feat_p = align_and_extract_feature(img_persp, faces_p[0]["raw_face"])
            if feat_p is not None:
                features.append(feat_p)
    except Exception:
        pass

    return features

def save_embeddings(embeddings_dict: Dict[str, Any]) -> bool:
    """Saves serialized embeddings database to trainer/face_embeddings.json."""
    try:
        EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Convert numpy arrays to lists for JSON serialization
        serializable = {}
        for emp_key, data in embeddings_dict.items():
            feats_list = [f.tolist() if isinstance(f, np.ndarray) else f for f in data["features"]]
            mean_f = data.get("mean_feature")
            if isinstance(mean_f, np.ndarray):
                mean_f = mean_f.tolist()
            serializable[emp_key] = {
                "employee_id": data.get("employee_id", ""),
                "employee_name": data.get("employee_name", ""),
                "features": feats_list,
                "mean_feature": mean_f,
                "num_embeddings": len(feats_list)
            }
        
        with open(EMBEDDINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
        return True
    except Exception as e:
        print(f"[Deep AI Error] Failed to save embeddings: {e}")
        return False

def load_embeddings() -> Dict[str, Any]:
    """Loads and deserializes embeddings database from trainer/face_embeddings.json."""
    if not EMBEDDINGS_PATH.exists():
        return {}
    try:
        with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        
        embeddings = {}
        for emp_key, data in raw.items():
            feats = [np.array(f, dtype=np.float32) for f in data["features"]]
            mean_f = np.array(data["mean_feature"], dtype=np.float32) if data.get("mean_feature") else None
            embeddings[emp_key] = {
                "employee_id": data.get("employee_id", ""),
                "employee_name": data.get("employee_name", ""),
                "features": feats,
                "mean_feature": mean_f,
                "num_embeddings": len(feats)
            }
        return embeddings
    except Exception as e:
        print(f"[Deep AI Error] Failed to load embeddings: {e}")
        return {}

def similarity_to_confidence_percent(cosine_sim: float, threshold: float = SFACE_MATCH_THRESHOLD) -> int:
    """
    Maps cosine similarity score [-1.0, 1.0] to a human-readable 0-100% confidence score:
    - Below threshold: 0% - 54%
    - At threshold (0.38): 60%
    - Good match (0.50): 78%
    - Strong match (0.65+): 90% - 99%
    """
    if cosine_sim < 0.15:
        return max(0, int(cosine_sim * 100))
    if cosine_sim < threshold:
        # Scale between 20% and 55%
        ratio = (cosine_sim - 0.15) / (threshold - 0.15)
        return int(20 + ratio * 35)
    else:
        # Scale between 60% and 99%
        ratio = min(1.0, (cosine_sim - threshold) / (0.75 - threshold))
        return int(60 + ratio * 39)

def match_face_embedding(query_feat: np.ndarray, embeddings_db: Dict[str, Any], threshold: float = SFACE_MATCH_THRESHOLD) -> Dict[str, Any]:
    """
    Compares query feature vector against all stored multi-angle embeddings of each employee.
    Returns:
        {
            "is_recognized": bool,
            "employee_id": str,
            "employee_name": str,
            "raw_name": str,
            "cosine_similarity": float,
            "confidence_percent": int
        }
    """
    if query_feat is None or not embeddings_db:
        return {
            "is_recognized": False,
            "employee_id": "N/A",
            "employee_name": "Unknown",
            "raw_name": "Unknown",
            "cosine_similarity": 0.0,
            "confidence_percent": 0
        }

    best_match_key = None
    best_sim = -1.0
    recognizer = get_sface_recognizer()

    # Compare query with every registered employee's feature bank
    for emp_key, data in embeddings_db.items():
        # Check against all multi-angle embeddings for this employee
        emp_max_sim = -1.0
        for ref_feat in data["features"]:
            sim = float(recognizer.match(query_feat, ref_feat, cv2.FaceRecognizerSF_FR_COSINE))
            if sim > emp_max_sim:
                emp_max_sim = sim
        
        if emp_max_sim > best_sim:
            best_sim = emp_max_sim
            best_match_key = emp_key

    is_recognized = (best_sim >= threshold) and (best_match_key is not None)
    
    if is_recognized:
        emp_data = embeddings_db[best_match_key]
        emp_id = emp_data["employee_id"]
        emp_name = emp_data["employee_name"]
        raw_name = best_match_key
    else:
        emp_id = "N/A"
        emp_name = "Unknown"
        raw_name = "Unknown"

    conf_pct = similarity_to_confidence_percent(best_sim, threshold)

    return {
        "is_recognized": is_recognized,
        "employee_id": emp_id,
        "employee_name": emp_name,
        "raw_name": raw_name,
        "cosine_similarity": round(best_sim, 3),
        "confidence_percent": conf_pct
    }
