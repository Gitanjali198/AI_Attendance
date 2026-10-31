"""
FastAPI REST API server for AI Attendance System.
Provides endpoints for employee management, attendance records, analytics, and model status.
"""
import sys
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.database import get_connection, init_db
from backend.employee import add_employee, get_all_employees, get_employee, delete_employee, update_employee
from backend.attendance import mark_attendance, record_manual_attendance, delete_attendance_record, get_today_summary
from backend.models import (
    EmployeeCreate, EmployeeResponse,
    AttendanceMarkRequest, AttendanceManualRequest,
    AttendanceRecordResponse, TodaySummaryResponse
)

# Initialize database
init_db()

app = FastAPI(
    title="📷 AI Face Attendance System API",
    description="High-performance REST API for face recognition attendance, employees, and analytics.",
    version="2.0.0"
)

# Enable CORS for cross-origin web/mobile apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Health"])
def root():
    return {
        "status": "online",
        "service": "AI Face Attendance System API",
        "version": "2.0.0",
        "docs_url": "/docs"
    }

# ----------------- EMPLOYEE ENDPOINTS -----------------

@app.get("/api/employees", response_model=List[EmployeeResponse], tags=["Employees"])
def list_employees():
    """Retrieve all registered employees."""
    return get_all_employees()

@app.get("/api/employees/{employee_id}", response_model=EmployeeResponse, tags=["Employees"])
def retrieve_employee(employee_id: str):
    """Retrieve an employee by their ID."""
    emp = get_employee(employee_id)
    if not emp:
        raise HTTPException(status_code=404, detail=f"Employee '{employee_id}' not found.")
    return emp

@app.post("/api/employees", tags=["Employees"])
def create_employee(payload: EmployeeCreate):
    """Register a new employee."""
    success, msg = add_employee(
        payload.employee_id,
        payload.employee_name,
        payload.department,
        payload.email or ""
    )
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"success": True, "message": msg, "employee_id": payload.employee_id}

@app.put("/api/employees/{employee_id}", tags=["Employees"])
def modify_employee(employee_id: str, payload: EmployeeCreate):
    """Update employee details."""
    success, msg = update_employee(
        employee_id,
        payload.employee_name,
        payload.department,
        payload.email or ""
    )
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"success": True, "message": msg}

@app.delete("/api/employees/{employee_id}", tags=["Employees"])
def remove_employee(employee_id: str, delete_images: bool = True):
    """Delete an employee and their face images."""
    success, msg = delete_employee(employee_id, delete_images=delete_images)
    if not success:
        raise HTTPException(status_code=404, detail=msg)
    return {"success": True, "message": msg}

# ----------------- ATTENDANCE ENDPOINTS -----------------

@app.get("/api/attendance", tags=["Attendance"])
def list_attendance(
    date: Optional[str] = Query(None, description="Filter by YYYY-MM-DD"),
    employee_id: Optional[str] = Query(None, description="Filter by Employee ID"),
    limit: int = Query(100, description="Max records to return")
):
    """Retrieve attendance records with optional filtering."""
    conn = get_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM attendance WHERE 1=1"
    params = []

    if date:
        query += " AND date = ?"
        params.append(date)
    if employee_id:
        query += " AND employee_id = ?"
        params.append(employee_id)

    query += " ORDER BY date DESC, id DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    records = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return {"count": len(records), "records": records}

@app.post("/api/attendance/mark", tags=["Attendance"])
def scan_mark_attendance(payload: AttendanceMarkRequest):
    """
    Mark attendance via face scan or manual trigger (Check-In or Check-Out).
    """
    result = mark_attendance(
        payload.employee_identifier,
        mode=payload.mode,
        custom_time=payload.custom_time,
        notes=payload.notes or ""
    )
    return result

@app.post("/api/attendance/manual", tags=["Attendance"])
def manual_attendance_entry(payload: AttendanceManualRequest):
    """Manually insert or override an attendance record for an employee."""
    success, msg = record_manual_attendance(
        payload.employee_id,
        payload.date,
        payload.check_in,
        payload.check_out or "",
        status=payload.status or "Present",
        notes=payload.notes or "Manual Entry"
    )
    return {"success": success, "message": msg}

@app.delete("/api/attendance/{record_id}", tags=["Attendance"])
def remove_attendance(record_id: int):
    """Delete an attendance record."""
    delete_attendance_record(record_id)
    return {"success": True, "message": f"Attendance record #{record_id} deleted."}

# ----------------- ANALYTICS & SUMMARY ENDPOINTS -----------------

@app.get("/api/summary/today", response_model=TodaySummaryResponse, tags=["Analytics"])
def get_daily_summary():
    """Retrieve today's attendance summary and KPIs."""
    return get_today_summary()

@app.get("/api/summary/departments", tags=["Analytics"])
def get_department_summary():
    """Retrieve employee counts and today's attendance grouped by department."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT department, COUNT(*) as total_employees
        FROM employees
        GROUP BY department
    """)
    departments = [dict(r) for r in cursor.fetchall()]

    conn.close()
    return {"departments": departments}

# ----------------- PAYROLL ENDPOINTS -----------------

@app.get("/api/payroll/calculate", tags=["Payroll"])
def get_payroll_calculation(
    start_date: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="End date YYYY-MM-DD")
):
    """Calculates payroll for all employees based on attendance."""
    from backend.payroll import calculate_all_payroll
    return calculate_all_payroll(start_date=start_date, end_date=end_date)

@app.get("/api/payroll/employee/{employee_id}", tags=["Payroll"])
def get_single_employee_payroll(
    employee_id: str,
    start_date: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="End date YYYY-MM-DD")
):
    """Calculates payroll and itemized days for a single employee."""
    from backend.payroll import calculate_employee_payroll
    result = calculate_employee_payroll(employee_id, start_date=start_date, end_date=end_date)
    if not result:
        raise HTTPException(status_code=404, detail=f"Employee '{employee_id}' not found.")
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
