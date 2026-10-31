"""
Employee management service for AI Attendance System.
Handles CRUD operations for employees, data retrieval, image folder management, and base salary.
"""
import sys
import shutil
from pathlib import Path
from datetime import datetime

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.database import get_connection
from backend.config import DATASET_DIR, DEFAULT_DAILY_SALARY, DEFAULT_HOURLY_RATE, DAILY_WORK_HOURS

def add_employee(employee_id, employee_name, department, email="", daily_salary=None, hourly_rate=DEFAULT_HOURLY_RATE):
    """
    Registers a new employee into the database.
    Returns (True, message) or (False, error_message).
    """
    employee_id = str(employee_id).strip()
    employee_name = str(employee_name).strip()
    department = str(department).strip()
    email = str(email).strip()
    try:
        hourly_rate = float(hourly_rate or DEFAULT_HOURLY_RATE)
    except Exception:
        hourly_rate = DEFAULT_HOURLY_RATE

    try:
        daily_salary = float(daily_salary) if daily_salary is not None else round(hourly_rate * DAILY_WORK_HOURS, 2)
    except Exception:
        daily_salary = round(hourly_rate * DAILY_WORK_HOURS, 2)

    if not employee_id or not employee_name or not department:
        return False, "Employee ID, Name, and Department are all required."

    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT employee_id FROM employees WHERE employee_id = ?", (employee_id,))
        if cursor.fetchone():
            conn.close()
            return False, f"Employee with ID '{employee_id}' already exists."

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO employees (employee_id, employee_name, department, email, created_at, daily_salary, hourly_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (employee_id, employee_name, department, email, now_str, daily_salary, hourly_rate))

        conn.commit()
        conn.close()
        return True, f"Employee '{employee_name}' registered successfully."
    except Exception as e:
        conn.close()
        return False, f"Database error: {str(e)}"

def get_all_employees():
    """Retrieves all registered employees as a list of dictionaries."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM employees ORDER BY employee_name ASC")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

def get_employee(employee_id):
    """Retrieves a single employee by employee_id."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM employees WHERE employee_id = ?", (str(employee_id).strip(),))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def update_employee(employee_id, employee_name, department, email="", daily_salary=None, hourly_rate=DEFAULT_HOURLY_RATE):
    """Updates an existing employee's details."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        try:
            hourly_rate = float(hourly_rate or DEFAULT_HOURLY_RATE)
        except Exception:
            hourly_rate = DEFAULT_HOURLY_RATE

        try:
            daily_salary = float(daily_salary) if daily_salary is not None else round(hourly_rate * DAILY_WORK_HOURS, 2)
        except Exception:
            daily_salary = round(hourly_rate * DAILY_WORK_HOURS, 2)

        cursor.execute("""
            UPDATE employees
            SET employee_name = ?, department = ?, email = ?, daily_salary = ?, hourly_rate = ?
            WHERE employee_id = ?
        """, (str(employee_name).strip(), str(department).strip(), str(email).strip(), daily_salary, hourly_rate, str(employee_id).strip()))
        conn.commit()
        conn.close()
        return True, "Employee updated successfully."
    except Exception as e:
        conn.close()
        return False, f"Error updating employee: {str(e)}"

def delete_employee(employee_id, delete_images=True, delete_attendance=False):
    """
    Deletes an employee, optionally removing their face dataset and attendance history.
    """
    employee_id = str(employee_id).strip()
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT employee_name FROM employees WHERE employee_id = ?", (employee_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return False, "Employee not found."

    emp_name = row["employee_name"]

    try:
        cursor.execute("DELETE FROM employees WHERE employee_id = ?", (employee_id,))
        if delete_attendance:
            cursor.execute("DELETE FROM attendance WHERE employee_id = ?", (employee_id,))
            cursor.execute("DELETE FROM payroll_records WHERE employee_id = ?", (employee_id,))

        conn.commit()
        conn.close()

        if delete_images:
            folder_pattern = f"{employee_id}_{emp_name}"
            target_folder = DATASET_DIR / folder_pattern
            if target_folder.exists() and target_folder.is_dir():
                shutil.rmtree(target_folder, ignore_errors=True)
            for p in DATASET_DIR.glob(f"{employee_id}_*"):
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)

        return True, f"Employee {emp_name} ({employee_id}) deleted successfully."
    except Exception as e:
        conn.close()
        return False, f"Error deleting employee: {str(e)}"

def get_departments():
    """Returns a list of distinct departments."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT department FROM employees WHERE department != '' ORDER BY department")
    depts = [r["department"] for r in cursor.fetchall()]
    conn.close()
    return depts
