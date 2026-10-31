"""
AI Attendance Management System - Unified CLI & Launcher
Provides single-command access to all subsystems:
- Streamlit Web Dashboard
- Live Face Recognition Kiosk
- Employee Registration & Face Dataset Collection
- AI Model Training
- FastAPI REST Backend Server
- Database Status Inspection
"""
import sys
import subprocess
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def run_dashboard():
    print("\n[Launcher] Starting Streamlit Web Dashboard...")
    cmd = [sys.executable, "-m", "streamlit", "run", str(BASE_DIR / "frontend" / "dashboard.py")]
    subprocess.run(cmd, cwd=str(BASE_DIR))

def run_kiosk():
    print("\n[Launcher] Launching Real-time Face Recognition Kiosk...")
    cmd = [sys.executable, "-m", "recognition.recognize_employee"]
    subprocess.run(cmd, cwd=str(BASE_DIR))

def run_register():
    print("\n[Launcher] Starting Employee Registration...")
    cmd = [sys.executable, "-m", "recognition.register_employee"]
    subprocess.run(cmd, cwd=str(BASE_DIR))

def run_train():
    print("\n[Launcher] Training AI Face Recognition Model...")
    from recognition.train_model import train_model
    res = train_model()
    if res["success"]:
        print(f"\n[SUCCESS] {res['message']}")
    else:
        print(f"\n[FAILED] {res['message']}")

def run_api():
    print("\n[Launcher] Starting FastAPI REST API at http://127.0.0.1:8000 (Docs: /docs)...")
    cmd = [sys.executable, "-m", "uvicorn", "backend.main:app", "--reload", "--host", "127.0.0.1", "--port", "8000"]
    subprocess.run(cmd, cwd=str(BASE_DIR))

def run_db_check():
    print("\n[Launcher] Checking Database State...")
    cmd = [sys.executable, str(BASE_DIR / "check_db.py")]
    subprocess.run(cmd, cwd=str(BASE_DIR))

def run_payroll():
    print("\n[Launcher] Calculating Current Month Payroll...")
    from backend.payroll import calculate_all_payroll
    res = calculate_all_payroll()
    print(f"\n=======================================================")
    print(f"   PAYROLL SUMMARY ({res['period_start']} to {res['period_end']})")
    print(f"   Wage Rate: Rs 100/hr | Calculated on Total Available Hours")
    print(f"=======================================================")
    print(f" Total Workforce:     {res['total_employees']}")
    print(f" Total Payroll Cost:  Rs {res['total_payroll_cost']:,.2f}")
    print(f" On-Time Days:        {res['total_full_days']}")
    print(f" Half Days (Late):    {res['total_half_days']}")
    print(f" Absent Days:         {res['total_absent_days']}")
    print(f" Total Hours Worked:  {res['total_hours']:.2f} hrs")
    print(f"-------------------------------------------------------")
    for e in res["employee_payrolls"]:
        print(f" • {e['employee_name']} (ID: {e['employee_id']}) -> Hours: {e['total_hours_worked']}h | Rate: Rs {e.get('hourly_rate', 100):.0f}/hr | Net: Rs {e['net_salary']:,.2f}")
    print(f"=======================================================\n")

def interactive_menu():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    while True:
        print("\n" + "=" * 55)
        print("    📷 AI ATTENDANCE & PAYROLL SYSTEM 2.0")
        print("=" * 55)
        print(" 1. 📊 Launch Web Portal Dashboard (Streamlit)")
        print(" 2. 📸 Launch Live Face Recognition Kiosk (OpenCV)")
        print(" 3. 👤 Register New Employee & Capture Face")
        print(" 4. 🧠 Train / Retrain AI Face Recognition Model")
        print(" 5. 💼 View Payroll & Salary Calculation")
        print(" 6. 🌐 Launch FastAPI REST Server (Swagger Docs)")
        print(" 7. 🗄️ Inspect SQLite Database")
        print(" 0. ❌ Exit")
        print("=" * 55)

        choice = input("Enter choice [1-7, 0 to exit]: ").strip()

        if choice == "1":
            run_dashboard()
        elif choice == "2":
            run_kiosk()
        elif choice == "3":
            run_register()
        elif choice == "4":
            run_train()
        elif choice == "5":
            run_payroll()
        elif choice == "6":
            run_api()
        elif choice == "7":
            run_db_check()
        elif choice in ["0", "q", "exit"]:
            print("Exiting. Goodbye!")
            break
        else:
            print("[Warning] Invalid selection. Please choose a number between 0 and 7.")

def main():
    parser = argparse.ArgumentParser(description="AI Attendance & Payroll System Launcher")
    parser.add_argument("-d", "--dashboard", action="store_true", help="Launch Streamlit Web Dashboard")
    parser.add_argument("-k", "--kiosk", action="store_true", help="Launch Live Face Recognition Kiosk")
    parser.add_argument("-r", "--register", action="store_true", help="Register New Employee")
    parser.add_argument("-t", "--train", action="store_true", help="Train AI Model")
    parser.add_argument("-p", "--payroll", action="store_true", help="View Payroll Summary")
    parser.add_argument("-a", "--api", action="store_true", help="Launch FastAPI REST Backend")
    parser.add_argument("-c", "--check", action="store_true", help="Check Database Status")

    args = parser.parse_args()

    if args.dashboard:
        run_dashboard()
    elif args.kiosk:
        run_kiosk()
    elif args.register:
        run_register()
    elif args.train:
        run_train()
    elif args.payroll:
        run_payroll()
    elif args.api:
        run_api()
    elif args.check:
        run_db_check()
    else:
        interactive_menu()

if __name__ == "__main__":
    main()
