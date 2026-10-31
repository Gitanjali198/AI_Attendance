"""
Database management and migration for AI Attendance System.
Handles table creation, schema migrations, and connection helpers.
"""
import sqlite3
import sys
import re
from pathlib import Path
from datetime import datetime

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.config import (
    DB_PATH, DEFAULT_HOURLY_RATE, DEFAULT_DAILY_SALARY, HALF_DAY_FACTOR, CHECKIN_GRACE_TIME
)

def get_connection():
    """Returns a SQLite connection with row factory enabled."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def calculate_duration(check_in_str, check_out_str):
    """Calculates formatted duration string between check_in and check_out times."""
    if not check_in_str or not check_out_str:
        return ""
    try:
        t_in = datetime.strptime(check_in_str.strip()[:8], "%H:%M:%S")
        t_out = datetime.strptime(check_out_str.strip()[:8], "%H:%M:%S")
        delta = t_out - t_in
        total_seconds = int(delta.total_seconds())
        if total_seconds < 0:
            return ""
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        if hours > 0:
            return f"{hours}h {minutes:02d}m {seconds:02d}s"
        return f"{minutes}m {seconds:02d}s"
    except Exception:
        return ""

def calculate_hours_between(check_in_str, check_out_str):
    """Calculates exact decimal hours between check_in and check_out timestamps."""
    if not check_in_str or not check_out_str:
        return 0.0
    try:
        t_in = datetime.strptime(check_in_str.strip()[:8], "%H:%M:%S")
        t_out = datetime.strptime(check_out_str.strip()[:8], "%H:%M:%S")
        delta = t_out - t_in
        total_seconds = int(delta.total_seconds())
        if total_seconds < 0:
            return 0.0
        return round(total_seconds / 3600.0, 4)
    except Exception:
        return 0.0

def duration_str_to_hours(duration_str):
    """Converts a duration string like '8h 30m 10s' or '15m 20s' to total float hours."""
    if not duration_str or not isinstance(duration_str, str):
        return 0.0
    try:
        hours = 0.0
        h_match = re.search(r'(\d+)\s*h', duration_str)
        m_match = re.search(r'(\d+)\s*m', duration_str)
        s_match = re.search(r'(\d+)\s*s', duration_str)
        if h_match:
            hours += float(h_match.group(1))
        if m_match:
            hours += float(m_match.group(1)) / 60.0
        if s_match:
            hours += float(s_match.group(1)) / 3600.0
        return round(hours, 4)
    except Exception:
        return 0.0

def init_db():
    """Initializes and migrates the database tables."""
    conn = get_connection()
    cursor = conn.cursor()

    # 1. Employees table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            employee_id TEXT PRIMARY KEY,
            employee_name TEXT NOT NULL,
            department TEXT NOT NULL,
            email TEXT DEFAULT '',
            created_at TEXT DEFAULT '',
            daily_salary REAL DEFAULT 800.0,
            hourly_rate REAL DEFAULT 100.0
        )
    """)

    # 2. Attendance table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT NOT NULL,
            employee_name TEXT NOT NULL,
            date TEXT NOT NULL,
            check_in TEXT DEFAULT '',
            check_out TEXT DEFAULT '',
            duration TEXT DEFAULT '',
            status TEXT DEFAULT 'Present',
            notes TEXT DEFAULT '',
            salary_earned REAL DEFAULT 0.0
        )
    """)

    # 3. Payroll Records table (Historical finalized runs)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS payroll_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT NOT NULL,
            employee_name TEXT NOT NULL,
            department TEXT NOT NULL,
            period_start TEXT NOT NULL,
            period_end TEXT NOT NULL,
            month_year TEXT NOT NULL,
            daily_salary REAL NOT NULL DEFAULT 800.0,
            hourly_rate REAL NOT NULL DEFAULT 100.0,
            total_days INTEGER NOT NULL DEFAULT 0,
            full_days INTEGER NOT NULL DEFAULT 0,
            half_days INTEGER NOT NULL DEFAULT 0,
            absent_days INTEGER NOT NULL DEFAULT 0,
            total_hours TEXT NOT NULL DEFAULT '',
            gross_salary REAL NOT NULL DEFAULT 0.0,
            deductions REAL NOT NULL DEFAULT 0.0,
            net_salary REAL NOT NULL DEFAULT 0.0,
            status TEXT NOT NULL DEFAULT 'Paid',
            generated_at TEXT NOT NULL
        )
    """)

    # Check & migrate employees columns
    cursor.execute("PRAGMA table_info(employees)")
    emp_columns = [col[1] for col in cursor.fetchall()]
    if "email" not in emp_columns:
        cursor.execute("ALTER TABLE employees ADD COLUMN email TEXT DEFAULT ''")
    if "created_at" not in emp_columns:
        cursor.execute("ALTER TABLE employees ADD COLUMN created_at TEXT DEFAULT ''")
    if "daily_salary" not in emp_columns:
        cursor.execute(f"ALTER TABLE employees ADD COLUMN daily_salary REAL DEFAULT {DEFAULT_DAILY_SALARY}")
    if "hourly_rate" not in emp_columns:
        cursor.execute(f"ALTER TABLE employees ADD COLUMN hourly_rate REAL DEFAULT {DEFAULT_HOURLY_RATE}")

    # Set default hourly rate and 8-hour daily equivalent for existing records
    cursor.execute("UPDATE employees SET hourly_rate = ? WHERE hourly_rate IS NULL", (DEFAULT_HOURLY_RATE,))
    cursor.execute("UPDATE employees SET daily_salary = ? WHERE daily_salary IS NULL OR daily_salary = 500.0", (DEFAULT_DAILY_SALARY,))

    # Check & migrate payroll_records columns
    cursor.execute("PRAGMA table_info(payroll_records)")
    pay_columns = [col[1] for col in cursor.fetchall()]
    if "hourly_rate" not in pay_columns:
        cursor.execute(f"ALTER TABLE payroll_records ADD COLUMN hourly_rate REAL DEFAULT {DEFAULT_HOURLY_RATE}")

    # Check & migrate attendance columns
    cursor.execute("PRAGMA table_info(attendance)")
    att_columns = [col[1] for col in cursor.fetchall()]
    if "duration" not in att_columns:
        cursor.execute("ALTER TABLE attendance ADD COLUMN duration TEXT DEFAULT ''")
    if "notes" not in att_columns:
        cursor.execute("ALTER TABLE attendance ADD COLUMN notes TEXT DEFAULT ''")
    if "salary_earned" not in att_columns:
        cursor.execute("ALTER TABLE attendance ADD COLUMN salary_earned REAL DEFAULT 0.0")

    # Recalculate durations and hourly salary (Rs 100/hr * exact time available)
    cursor.execute("SELECT id, employee_id, check_in, check_out, duration, status, salary_earned FROM attendance")
    for row in cursor.fetchall():
        row_id = row["id"]
        check_in = row["check_in"] or ""
        check_out = row["check_out"] or ""
        duration = row["duration"] or ""
        curr_status = row["status"] or ""

        new_duration = duration
        if check_in and check_out:
            new_duration = calculate_duration(check_in, check_out)
            hours_avail = calculate_hours_between(check_in, check_out)
            # Salary = total hours * Rs 100/hr
            new_salary = round(hours_avail * DEFAULT_HOURLY_RATE, 2)
        elif check_in and not check_out:
            # Still checked in (no checkout yet)
            new_salary = 0.0
        else:
            new_salary = 0.0

        # Punctuality tagging rule: check-in after 09:30:00 is tagged Half Day
        new_status = curr_status
        if check_in:
            if check_in[:8] > CHECKIN_GRACE_TIME:
                new_status = "Half Day"
            elif curr_status in ["Present", ""]:
                new_status = "Present"

        cursor.execute("""
            UPDATE attendance
            SET duration = ?, status = ?, salary_earned = ?
            WHERE id = ?
        """, (new_duration, new_status, new_salary, row_id))

    conn.commit()
    conn.close()
    print("Database Initialized & Migrated Successfully to Hourly Rate (Rs 100/hr).")

if __name__ == "__main__":
    init_db()
