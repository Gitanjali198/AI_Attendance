"""
Database inspector script for AI Attendance System.
Displays tables, employees, and recent attendance records.
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.database import get_connection

conn = get_connection()
cursor = conn.cursor()

# Show tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [r["name"] for r in cursor.fetchall()]
print("=== DATABASE TABLES ===")
print(tables)

# Show employees
try:
    cursor.execute("SELECT * FROM employees")
    rows = cursor.fetchall()
    print(f"\n=== EMPLOYEES ({len(rows)}) ===")
    for row in rows:
        print(dict(row))
except Exception as e:
    print("Employees Query Error:", e)

# Show attendance
try:
    cursor.execute("SELECT * FROM attendance ORDER BY date DESC, id DESC LIMIT 10")
    rows = cursor.fetchall()
    print(f"\n=== RECENT ATTENDANCE RECORDS (showing up to 10) ===")
    for row in rows:
        print(dict(row))
except Exception as e:
    print("Attendance Query Error:", e)

conn.close()
