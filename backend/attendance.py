"""
Attendance recording and calculation service for AI Attendance System.
Handles real-time check-in, check-out, duration calculation, punctuality tagging,
half-day determination (after 9:30 AM), and salary calculation.
"""
import sys
from pathlib import Path
from datetime import datetime, date

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.database import get_connection, calculate_duration, calculate_hours_between, duration_str_to_hours
from backend.config import (
    WORK_START_TIME, CHECKIN_GRACE_TIME, WORK_END_TIME,
    CHECKOUT_GRACE_TIME, DEFAULT_HOURLY_RATE, DEFAULT_DAILY_SALARY, HALF_DAY_FACTOR
)

def parse_employee_identifier(employee_identifier):
    """
    Parses employee_id and employee_name from either:
    - '101_Simra Kazi' (folder / label name)
    - '101' (id alone)
    """
    conn = get_connection()
    cursor = conn.cursor()

    if "_" in str(employee_identifier):
        parts = str(employee_identifier).split("_", 1)
        emp_id = parts[0].strip()
        fallback_name = parts[1].strip()
    else:
        emp_id = str(employee_identifier).strip()
        fallback_name = emp_id

    # Query DB to get canonical employee name
    cursor.execute("SELECT employee_id, employee_name, department, daily_salary, hourly_rate FROM employees WHERE employee_id = ?", (emp_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        row_dict = dict(row)
        hourly_rate = float(row_dict.get("hourly_rate") or DEFAULT_HOURLY_RATE)
        daily_salary = float(row_dict.get("daily_salary") or (hourly_rate * 8.0))
        return row["employee_id"], row["employee_name"], row["department"], hourly_rate, daily_salary
    return emp_id, fallback_name, "General", DEFAULT_HOURLY_RATE, DEFAULT_DAILY_SALARY

def mark_attendance(employee_identifier, mode="checkin", custom_time=None, notes=""):
    """
    Marks attendance for an employee in either 'checkin' or 'checkout' mode.
    Enforces work shift:
    - Work hours: 9:00 AM to 5:00 PM (8 hrs)
    - Hourly rate: Rs 100 per hour
    - Salary calculated based on exact total hours employee is available in company (check-in to check-out)
    - Check-in: 9:00 to 9:30 AM is On-Time
    - Check-in after 9:30 AM is tagged Half Day
    - Check-out: 5:00 to 5:30 PM
    """
    emp_id, emp_name, department, hourly_rate, daily_salary = parse_employee_identifier(employee_identifier)

    today = date.today().strftime("%Y-%m-%d")
    current_time = custom_time or datetime.now().strftime("%H:%M:%S")

    conn = get_connection()
    cursor = conn.cursor()

    mode = mode.lower().strip()

    # ---------------- CHECK IN ----------------
    if mode in ["checkin", "check_in", "in"]:
        cursor.execute("""
            SELECT id, check_in, status FROM attendance
            WHERE employee_id = ? AND date = ?
        """, (emp_id, today))
        record = cursor.fetchone()

        if record:
            conn.close()
            return {
                "status": "already_checked_in",
                "message": f"{emp_name} ({emp_id}) already checked in today at {record['check_in']}.",
                "employee_id": emp_id,
                "employee_name": emp_name,
                "department": department,
                "time": record["check_in"],
                "date": today
            }

        # Check-in timing rule:
        # 9:00 - 9:30 AM -> On-Time
        # After 9:30 AM -> Tagged Half Day
        clean_time = current_time[:8]
        entry_notes = notes

        if clean_time <= CHECKIN_GRACE_TIME:
            status = "Present"
            if not entry_notes:
                entry_notes = f"On-Time Check-In (Shift 9:00 AM - 5:00 PM)"
            msg = f"Welcome, {emp_name}! Check-in recorded at {current_time}. Status: On-Time. Salary will be calculated at checkout at Rs {hourly_rate:.0f}/hr."
        else:
            status = "Half Day"
            if not entry_notes:
                entry_notes = f"Late Check-In (After {CHECKIN_GRACE_TIME[:5]} AM - Half Day tag)"
            msg = f"Welcome, {emp_name}! Check-in recorded at {current_time}. Marked as Half Day (After 09:30 AM). Salary will be calculated at checkout at Rs {hourly_rate:.0f}/hr."

        # At check-in, duration is 0, salary is pending checkout
        salary_earned = 0.0

        cursor.execute("""
            INSERT INTO attendance (
                employee_id, employee_name, date, check_in, check_out, duration, status, notes, salary_earned
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (emp_id, emp_name, today, current_time, "", "", status, entry_notes, salary_earned))

        conn.commit()
        conn.close()

        return {
            "status": "checkin_success",
            "message": msg,
            "employee_id": emp_id,
            "employee_name": emp_name,
            "department": department,
            "time": current_time,
            "date": today,
            "attendance_status": status,
            "salary_earned": salary_earned
        }

    # ---------------- CHECK OUT ----------------
    elif mode in ["checkout", "check_out", "out"]:
        cursor.execute("""
            SELECT id, check_in, check_out, status, notes, salary_earned FROM attendance
            WHERE employee_id = ? AND date = ?
        """, (emp_id, today))
        record = cursor.fetchone()

        if not record:
            conn.close()
            return {
                "status": "not_checked_in",
                "message": f"{emp_name} ({emp_id}) has not checked in today yet.",
                "employee_id": emp_id,
                "employee_name": emp_name,
                "department": department,
                "time": current_time,
                "date": today
            }

        attendance_id = record["id"]
        check_in_time = record["check_in"]
        existing_check_out = record["check_out"]
        curr_status = record["status"]
        curr_notes = record["notes"] or ""

        if existing_check_out and existing_check_out != "":
            conn.close()
            return {
                "status": "already_checked_out",
                "message": f"{emp_name} ({emp_id}) already checked out today at {existing_check_out}.",
                "employee_id": emp_id,
                "employee_name": emp_name,
                "department": department,
                "time": existing_check_out,
                "date": today
            }

        duration = calculate_duration(check_in_time, current_time)
        hours_worked = calculate_hours_between(check_in_time, current_time)
        # Calculate salary: Rs 100/hr * exact hours available between check-in and check-out
        curr_salary = round(hours_worked * hourly_rate, 2)
        clean_checkout = current_time[:8]

        # Check-out timing notes (5:00 - 5:30 PM window)
        checkout_note = ""
        if clean_checkout < WORK_END_TIME:
            checkout_note = f"Early Check-Out (Before {WORK_END_TIME[:5]} PM)"
        elif clean_checkout <= CHECKOUT_GRACE_TIME:
            checkout_note = f"On-Time Check-Out ({WORK_END_TIME[:5]} - {CHECKOUT_GRACE_TIME[:5]} PM)"
        else:
            checkout_note = f"Late Check-Out (After {CHECKOUT_GRACE_TIME[:5]} PM)"

        final_notes = f"{curr_notes} | {checkout_note}" if curr_notes else checkout_note

        cursor.execute("""
            UPDATE attendance
            SET check_out = ?, duration = ?, notes = ?, salary_earned = ?
            WHERE id = ?
        """, (current_time, duration, final_notes, curr_salary, attendance_id))

        conn.commit()
        conn.close()

        return {
            "status": "checkout_success",
            "message": f"Goodbye, {emp_name}! Checked out at {current_time}. Total Time: {duration or 'N/A'} ({hours_worked:.2f} hrs). Salary Earned: Rs {curr_salary:.2f} (at Rs {hourly_rate:.0f}/hr).",
            "employee_id": emp_id,
            "employee_name": emp_name,
            "department": department,
            "time": current_time,
            "date": today,
            "duration": duration,
            "hours_worked": hours_worked,
            "attendance_status": curr_status,
            "salary_earned": curr_salary
        }

    conn.close()
    return {
        "status": "error",
        "message": f"Invalid mode '{mode}'. Choose 'checkin' or 'checkout'.",
        "employee_id": emp_id,
        "employee_name": emp_name,
        "department": department,
        "time": current_time,
        "date": today
    }

def record_manual_attendance(employee_id, date_str, check_in, check_out, status="Present", notes="Manual Override"):
    """
    Inserts or updates an attendance record manually.
    Computes salary based on exact hours available between check_in and check_out at Rs 100/hr.
    """
    emp_id, emp_name, _, hourly_rate, daily_salary = parse_employee_identifier(employee_id)
    duration = calculate_duration(check_in, check_out)
    hours_worked = calculate_hours_between(check_in, check_out)

    # Determine salary earned: Total hours * Rs 100/hr
    if check_in and check_out:
        salary_earned = round(hours_worked * hourly_rate, 2)
    else:
        salary_earned = 0.0

    conn = get_connection()
    cursor = conn.cursor()

    # Check if record exists for this employee on this date
    cursor.execute("SELECT id FROM attendance WHERE employee_id = ? AND date = ?", (emp_id, date_str))
    existing = cursor.fetchone()

    if existing:
        cursor.execute("""
            UPDATE attendance
            SET check_in = ?, check_out = ?, duration = ?, status = ?, notes = ?, salary_earned = ?
            WHERE id = ?
        """, (check_in, check_out, duration, status, notes, salary_earned, existing["id"]))
    else:
        cursor.execute("""
            INSERT INTO attendance (
                employee_id, employee_name, date, check_in, check_out, duration, status, notes, salary_earned
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (emp_id, emp_name, date_str, check_in, check_out, duration, status, notes, salary_earned))

    conn.commit()
    conn.close()
    return True, f"Attendance record saved for {emp_name} on {date_str} ({status} - {hours_worked:.2f} hrs - Rs {salary_earned:.2f})."

def delete_attendance_record(record_id):
    """Deletes an attendance entry by its ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM attendance WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()
    return True

def get_today_summary():
    """
    Returns comprehensive KPI metrics for today including payroll stats.
    """
    today = date.today().strftime("%Y-%m-%d")
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as count FROM employees")
    total_employees = cursor.fetchone()["count"]

    cursor.execute("SELECT * FROM attendance WHERE date = ?", (today,))
    today_records = [dict(r) for r in cursor.fetchall()]

    conn.close()

    present_ids = {r["employee_id"] for r in today_records}
    present_count = len(present_ids)
    full_day_count = sum(1 for r in today_records if r.get("status") in ["Present", "Full Day"])
    half_day_count = sum(1 for r in today_records if r.get("status") in ["Half Day", "Half-Day"])
    checked_out_count = sum(1 for r in today_records if r.get("check_out"))
    currently_in_count = max(present_count - checked_out_count, 0)
    absent_count = max(total_employees - present_count, 0)
    attendance_rate = (present_count / total_employees * 100) if total_employees > 0 else 0.0
    today_salary_total = sum(float(r.get("salary_earned") or 0.0) for r in today_records)

    return {
        "date": today,
        "total_employees": total_employees,
        "present_count": present_count,
        "full_day_count": full_day_count,
        "half_day_count": half_day_count,
        "checked_out_count": checked_out_count,
        "currently_in_count": currently_in_count,
        "absent_count": absent_count,
        "attendance_rate": round(attendance_rate, 1),
        "today_salary_total": today_salary_total,
        "records": today_records
    }
