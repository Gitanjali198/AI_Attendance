"""
Database reset script for AI Attendance System.
Clears attendance records and resets auto-increment IDs.
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.database import get_connection

def reset_db(include_employees=False):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM attendance")
    cursor.execute("DELETE FROM sqlite_sequence WHERE name='attendance'")

    if include_employees:
        cursor.execute("DELETE FROM employees")
        print("Cleared employees and attendance records.")
    else:
        print("Cleared attendance records (kept employee profiles).")

    conn.commit()
    conn.close()
    print("Database cleaned successfully!")

if __name__ == "__main__":
    confirm = input("Reset attendance logs? Type 'yes' to proceed: ").strip().lower()
    if confirm in ["yes", "y"]:
        reset_db(include_employees=False)
    else:
        print("Operation cancelled.")
