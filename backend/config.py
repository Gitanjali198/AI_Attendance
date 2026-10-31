"""
Configuration and centralized paths for AI Attendance System.
Handles dynamic path resolution across all modules.
"""
from pathlib import Path
import os

# Base project directory (AI_Attendance_System-main root)
BASE_DIR = Path(__file__).resolve().parent.parent

# Database configuration (support cloud persistent disk mount via env var)
DATABASE_DIR = Path(os.getenv("DATABASE_DIR", BASE_DIR / "database"))
DB_PATH = Path(os.getenv("DB_PATH", DATABASE_DIR / "attendance.db"))

# Haar Cascade Face Detector
HAARCASCADE_DIR = Path(os.getenv("HAARCASCADE_DIR", BASE_DIR / "haarcascade"))
HAAR_CASCADE_PATH = HAARCASCADE_DIR / "haarcascade_frontalface_default.xml"

# Face Dataset Directory (support cloud persistent disk mount via env var)
DATASET_DIR = Path(os.getenv("DATASET_DIR", BASE_DIR / "employee_images"))

# Model Trainer Directory & Files (support cloud persistent disk mount via env var)
TRAINER_DIR = Path(os.getenv("TRAINER_DIR", BASE_DIR / "trainer"))
TRAINER_MODEL_PATH = TRAINER_DIR / "trainer.yml"
LABELS_PATH = TRAINER_DIR / "labels.json"

# Reports Output Directory
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", BASE_DIR / "reports"))

# Ensure all essential directories exist
for directory in [DATABASE_DIR, HAARCASCADE_DIR, DATASET_DIR, TRAINER_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Face Recognition Settings
DEFAULT_CONFIDENCE_THRESHOLD = 45  # Max LBPH distance allowed (distance <= 45 is a genuine match)
MIN_CONFIDENCE_PERCENT = 58        # Minimum match confidence percentage (>= 58%)
FACE_IMAGE_SIZE = (200, 200)
BURST_CAPTURE_COUNT = 50           # Images captured during registration
DEFAULT_CAMERA_INDEX = 0

# Work Shift & Timing Settings (9:00 AM to 5:00 PM, 8 hours)
WORK_START_TIME = "09:00:00"         # Shift starts at 9:00 AM
CHECKIN_GRACE_TIME = "09:30:00"      # On-Time window: 09:00 to 09:30. After 09:30 is marked Half Day
WORK_END_TIME = "17:00:00"           # Shift ends at 5:00 PM
CHECKOUT_GRACE_TIME = "17:30:00"     # Checkout window: 17:00 to 17:30
DAILY_WORK_HOURS = 8.0               # Standard 8 hours daily

# Payroll Settings - Hourly Rate Based System
DEFAULT_HOURLY_RATE = 100.0          # Rs 100 per hour
DEFAULT_DAILY_SALARY = 800.0         # Rs 800 for 8-hour shift (8 hrs * Rs 100)
HALF_DAY_FACTOR = 0.5                # Half day factor (4 hours = Rs 400)
CURRENCY_SYMBOL = "Rs."
