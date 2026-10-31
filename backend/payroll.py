"""
Payroll and Compensation Engine for AI Attendance System.
Calculates employee salaries based on daily work shift (9:00 AM - 5:00 PM, 8 hours),
punctuality rules (check-in before 9:30 AM = Full Day Rs 500, after 9:30 AM = Half Day Rs 250),
check-out timings (5:00 - 5:30 PM), working hours, and generates itemized pay slips.
"""
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.config import (
    WORK_START_TIME, CHECKIN_GRACE_TIME, WORK_END_TIME,
    CHECKOUT_GRACE_TIME, DAILY_WORK_HOURS, DEFAULT_HOURLY_RATE,
    DEFAULT_DAILY_SALARY, HALF_DAY_FACTOR, CURRENCY_SYMBOL
)
from backend.database import (
    get_connection, duration_str_to_hours, calculate_hours_between, calculate_duration
)
from backend.employee import get_all_employees, get_employee

def parse_date_range(start_date=None, end_date=None, month=None, year=None):
    """
    Parses start and end dates. If month/year provided, returns month range.
    Defaults to current month if nothing provided.
    """
    today = date.today()
    if month and year:
        # Month range
        start = date(year, month, 1)
        # Next month first day minus 1
        if month == 12:
            end = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            end = date(year, month + 1, 1) - timedelta(days=1)
        return start, end

    if start_date and end_date:
        if isinstance(start_date, str):
            start = datetime.strptime(start_date[:10], "%Y-%m-%d").date()
        else:
            start = start_date

        if isinstance(end_date, str):
            end = datetime.strptime(end_date[:10], "%Y-%m-%d").date()
        else:
            end = end_date
        return start, end

    # Default to current month
    start = date(today.year, today.month, 1)
    end = today
    return start, end

def calculate_employee_payroll(employee_id, start_date=None, end_date=None, month=None, year=None, custom_hourly_rate=None, custom_daily_rate=None):
    """
    Calculates detailed attendance and payroll for a single employee over a date range.
    Calculates salary as Rs 100/hr multiplied by exact hours available in company (check-in to check-out).
    Returns itemized daily records and payroll totals.
    """
    emp = get_employee(employee_id)
    if not emp:
        return None

    start, end = parse_date_range(start_date, end_date, month, year)
    hourly_rate = float(custom_hourly_rate or emp.get("hourly_rate") or DEFAULT_HOURLY_RATE)
    daily_rate = float(custom_daily_rate or emp.get("daily_salary") or (hourly_rate * DAILY_WORK_HOURS))
    half_day_rate = round(daily_rate * HALF_DAY_FACTOR, 2)

    conn = get_connection()
    cursor = conn.cursor()

    # Query attendance records in range
    cursor.execute("""
        SELECT * FROM attendance
        WHERE employee_id = ? AND date BETWEEN ? AND ?
        ORDER BY date ASC, id ASC
    """, (str(employee_id).strip(), start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
    att_rows = [dict(r) for r in cursor.fetchall()]
    conn.close()

    # Map attendance records by date
    att_by_date = {}
    for r in att_rows:
        att_by_date[r["date"]] = r

    # Iterate through all dates in range
    today = date.today()
    curr_date = start
    daily_records = []
    full_days = 0
    half_days = 0
    absent_days = 0
    total_hours = 0.0

    while curr_date <= end:
        date_str = curr_date.strftime("%Y-%m-%d")
        record = att_by_date.get(date_str)

        # Check if record exists
        if record:
            check_in = record.get("check_in", "")
            check_out = record.get("check_out", "")
            status = record.get("status", "Present")
            duration_str = record.get("duration", "")
            notes = record.get("notes", "")

            # Calculate exact hours available between check-in and check-out
            if check_in and check_out:
                day_hours = calculate_hours_between(check_in, check_out)
                if day_hours == 0.0 and duration_str:
                    day_hours = duration_str_to_hours(duration_str)
                earned = round(day_hours * hourly_rate, 2)
            elif check_in and not check_out:
                day_hours = 0.0
                earned = 0.0
                duration_str = "In Progress"
            else:
                day_hours = 0.0
                earned = 0.0

            total_hours += day_hours

            # Determine whether Full Day or Half Day according to user rules:
            # Rule 1: Check in 9:00 - 9:30 AM is On-Time.
            # Rule 2: Check in after 9:30 AM is Half Day.
            is_half_day = False
            if check_in:
                check_in_clean = check_in.strip()[:8]
                if check_in_clean > CHECKIN_GRACE_TIME:
                    is_half_day = True
            if status in ["Half Day", "Half-Day"]:
                is_half_day = True

            if is_half_day:
                half_days += 1
                day_status = "Half Day"
            else:
                full_days += 1
                day_status = "Full Day (Present)"

            daily_records.append({
                "date": date_str,
                "day_name": curr_date.strftime("%a"),
                "check_in": check_in or "-",
                "check_out": check_out or "-",
                "duration": duration_str or "-",
                "hours": round(day_hours, 2),
                "status": day_status,
                "salary_earned": earned,
                "notes": notes or ("After 09:30 AM" if is_half_day else "On Time")
            })

        else:
            # No attendance record
            # Only count as absent if date is today or in the past
            if curr_date <= today:
                absent_days += 1
                daily_records.append({
                    "date": date_str,
                    "day_name": curr_date.strftime("%a"),
                    "check_in": "-",
                    "check_out": "-",
                    "duration": "-",
                    "hours": 0.0,
                    "status": "Absent",
                    "salary_earned": 0.0,
                    "notes": "No attendance recorded"
                })
            else:
                # Future date in the period
                daily_records.append({
                    "date": date_str,
                    "day_name": curr_date.strftime("%a"),
                    "check_in": "-",
                    "check_out": "-",
                    "duration": "-",
                    "hours": 0.0,
                    "status": "Upcoming",
                    "salary_earned": 0.0,
                    "notes": "Upcoming scheduled date"
                })

        curr_date += timedelta(days=1)

    # Daily Net Salary = Hours worked * 100 for per day
    # Monthly Net Salary = Total of all days' net salary
    total_month_net_salary = round(sum(float(r.get("salary_earned", 0.0) or 0.0) for r in daily_records), 2)
    gross_salary = total_month_net_salary
    deductions = 0.0
    net_salary = total_month_net_salary

    total_seconds_int = int(round(total_hours * 3600))
    hours_int = total_seconds_int // 3600
    minutes_int = (total_seconds_int % 3600) // 60
    formatted_hours = f"{hours_int}h {minutes_int:02d}m"

    return {
        "employee_id": emp["employee_id"],
        "employee_name": emp["employee_name"],
        "department": emp["department"],
        "email": emp.get("email", ""),
        "hourly_rate": hourly_rate,
        "daily_salary": daily_rate,
        "half_day_rate": half_day_rate,
        "period_start": start.strftime("%Y-%m-%d"),
        "period_end": end.strftime("%Y-%m-%d"),
        "total_days_evaluated": len(daily_records),
        "full_days": full_days,
        "half_days": half_days,
        "absent_days": absent_days,
        "total_hours_worked": round(total_hours, 2),
        "formatted_hours": formatted_hours,
        "gross_salary": gross_salary,
        "deductions": deductions,
        "net_salary": net_salary,
        "daily_records": daily_records
    }

def calculate_all_payroll(start_date=None, end_date=None, month=None, year=None):
    """
    Calculates payroll summary for all registered employees.
    """
    start, end = parse_date_range(start_date, end_date, month, year)
    employees = get_all_employees()

    employee_payrolls = []
    total_payroll_cost = 0.0
    total_full_days = 0
    total_half_days = 0
    total_absent_days = 0
    total_hours_all = 0.0

    for emp in employees:
        pay_info = calculate_employee_payroll(
            emp["employee_id"],
            start_date=start,
            end_date=end
        )
        if pay_info:
            employee_payrolls.append(pay_info)
            total_payroll_cost += pay_info["net_salary"]
            total_full_days += pay_info["full_days"]
            total_half_days += pay_info["half_days"]
            total_absent_days += pay_info["absent_days"]
            total_hours_all += pay_info["total_hours_worked"]

    return {
        "period_start": start.strftime("%Y-%m-%d"),
        "period_end": end.strftime("%Y-%m-%d"),
        "total_employees": len(employees),
        "total_payroll_cost": total_payroll_cost,
        "total_full_days": total_full_days,
        "total_half_days": total_half_days,
        "total_absent_days": total_absent_days,
        "total_hours": round(total_hours_all, 2),
        "employee_payrolls": employee_payrolls
    }

def save_finalized_payroll(payroll_data):
    """
    Saves or updates a finalized payroll run in the payroll_records table.
    """
    conn = get_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    saved_count = 0
    for record in payroll_data.get("employee_payrolls", []):
        cursor.execute("""
            INSERT INTO payroll_records (
                employee_id, employee_name, department, period_start, period_end,
                month_year, daily_salary, hourly_rate, total_days, full_days, half_days, absent_days,
                total_hours, gross_salary, deductions, net_salary, status, generated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record["employee_id"],
            record["employee_name"],
            record["department"],
            record["period_start"],
            record["period_end"],
            datetime.strptime(record["period_start"], "%Y-%m-%d").strftime("%B %Y"),
            record["daily_salary"],
            record.get("hourly_rate", DEFAULT_HOURLY_RATE),
            record["total_days_evaluated"],
            record["full_days"],
            record["half_days"],
            record["absent_days"],
            record["formatted_hours"],
            record["gross_salary"],
            record["deductions"],
            record["net_salary"],
            "Paid",
            now_str
        ))
        saved_count += 1

    conn.commit()
    conn.close()
    return saved_count
