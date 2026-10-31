"""
Pydantic data schemas for FastAPI REST API endpoints.
"""
from pydantic import BaseModel, Field
from typing import Optional, List

class EmployeeBase(BaseModel):
    employee_id: str = Field(..., description="Unique alphanumeric Employee ID")
    employee_name: str = Field(..., description="Full Name of Employee")
    department: str = Field(..., description="Department or Team")
    email: Optional[str] = Field("", description="Work Email Address")
    daily_salary: Optional[float] = Field(800.0, description="Base Daily Salary for 8h shift (Rs)")
    hourly_rate: Optional[float] = Field(100.0, description="Base Hourly Wage Rate (Rs/hr)")

class EmployeeCreate(EmployeeBase):
    pass

class EmployeeResponse(EmployeeBase):
    created_at: Optional[str] = ""

class AttendanceMarkRequest(BaseModel):
    employee_identifier: str = Field(..., description="Employee ID or ID_Name")
    mode: str = Field("checkin", description="'checkin' or 'checkout'")
    custom_time: Optional[str] = Field(None, description="Optional custom timestamp HH:MM:SS")
    notes: Optional[str] = Field("", description="Optional notes or tag")

class AttendanceManualRequest(BaseModel):
    employee_id: str
    date: str = Field(..., description="YYYY-MM-DD")
    check_in: str = Field(..., description="HH:MM:SS")
    check_out: Optional[str] = Field("", description="HH:MM:SS")
    status: Optional[str] = Field("Present", description="Present, Half Day, etc.")
    notes: Optional[str] = Field("Manual Entry", description="Reason or notes")

class AttendanceRecordResponse(BaseModel):
    id: int
    employee_id: str
    employee_name: str
    date: str
    check_in: str
    check_out: str
    duration: str
    status: str
    notes: str
    salary_earned: Optional[float] = 0.0

class TodaySummaryResponse(BaseModel):
    date: str
    total_employees: int
    present_count: int
    full_day_count: int
    half_day_count: int
    checked_out_count: int
    currently_in_count: int
    absent_count: int
    attendance_rate: float
    today_salary_total: float

class PayrollEmployeeResponse(BaseModel):
    employee_id: str
    employee_name: str
    department: str
    hourly_rate: float
    daily_salary: float
    period_start: str
    period_end: str
    total_days_evaluated: int
    full_days: int
    half_days: int
    absent_days: int
    total_hours_worked: float
    formatted_hours: str
    gross_salary: float
    net_salary: float
