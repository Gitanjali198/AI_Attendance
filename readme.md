# 📷 AI Attendance Management System 2.0 (Computer Vision & Biometrics)

A modern, enterprise-grade AI Attendance System powered by **OpenCV Facial Recognition**, **Streamlit Web Portal**, **SQLite Database**, and **FastAPI REST API**.

The system automates employee attendance tracking via face recognition, calculates working hours, tags punctuality (On-Time / Late), and provides an interactive web dashboard with analytics, employee management, and exportable payroll reports.

---

## 🌟 Key Features & Enhancements

### 1. 💼 Automated Attendance-Based Payroll Engine
- **Work Shift Policy**: 8 hours daily, from **9:00 AM to 5:00 PM**.
- **On-Time Check-In Window**: 9:00 AM to 9:30 AM $\rightarrow$ Marked as **Full Day (100% Daily Wage = ₹500)**.
- **Late Check-In Rule**: Check-in after 9:30 AM $\rightarrow$ Automatically marked as **Half Day (50% Daily Wage = ₹250)**.
- **Check-Out Window**: 5:00 PM to 5:30 PM (completed 8-hour shift).
- **Salary Computation**: Automatically computes total gross and net salary for all registered employees based on Full Days ($N \times \text{₹}500$) and Half Days ($M \times \text{₹}250$).
- **Official Payslip Voucher Generator**: Generates clean, downloadable, and printable HTML Salary Slips for each employee.
- **Payroll Reports Export**: One-click download of the entire organization's payroll sheet as **Excel (.xlsx)** and **CSV**.

### 2. 📊 Executive Web Dashboard (Streamlit 2.0)
- **Real-Time KPIs**: Total Workforce, Today's Present, In-Office Active, Checked Out, Absent, and Attendance Rate (%).
- **Interactive Visualizations (Plotly)**:
  - Donut chart of attendance breakdown.
  - Stacked department attendance chart.
  - Historical multi-day attendance trends.
- **Dynamic Period Filters**: Today, Yesterday, Last 7 Days, Last 30 Days, Custom Date Range, or All-Time.
- **One-Click Export**: Download professional **CSV** and **Excel (.xlsx)** reports ready for payroll.

### 2. 📸 Dual-Mode Attendance Recognition
- **Option A: In-Browser Web Camera**: Mark attendance directly from any web browser using `st.camera_input` with instant face bounding box, identity matching, confidence percentage, and audio confirmation.
- **Option B: Real-Time Desktop Live Kiosk (30 FPS)**:
  - High-speed OpenCV camera window with HUD overlay.
  - Keyboard hotkey `m` to toggle between **Check-In** and **Check-Out** on the fly.
  - Anti-spam cooldown timer to prevent redundant duplicate records.
  - Audio chime feedback on successful face verification.

### 3. 👥 Employee Management & Face Dataset Studio
- **Employee Directory**: Searchable list of employees with department filters and biometric sample previews.
- **Multi-Method Biometric Registration**:
  - **Photo Upload**: Drag and drop existing face photos (`.jpg`, `.png`).
  - **Webcam Burst Capture**: Automatically crops, normalizes, and saves face samples to `employee_images/{id}_{name}/`.
- **Edit & Delete**: Safely manage employee profiles with automatic dataset cleanup.

### 4. 🧠 1-Click AI Model Trainer
- Retrain the OpenCV **LBPH (Local Binary Patterns Histograms)** face recognizer with one click from the UI or command line.
- Live dataset explorer to inspect captured face crops for each employee.
- Model diagnostics (trained class count, model size, last trained timestamp).

### 5. ⏱️ Working Hours & Punctuality Engine
- **Automated Work Duration**: Computes total hours and minutes worked between Check-In and Check-Out.
- **Late Arrival Tagging**: Automatically flags arrivals after office start time (e.g., 09:30 AM).
- **Manual Attendance Overrides**: HR can manually record or adjust attendance entries with custom notes.

### 6. 🌐 FastAPI REST Backend
- Complete REST API with Swagger documentation at `http://127.0.0.1:8000/docs`.
- Clean endpoints for mobile apps or external HR systems integration.

---

## 📂 Project Architecture

```
AI_Attendance_System-main/
│
├── backend/
│   ├── config.py              # Centralized paths and system constants
│   ├── database.py            # SQLite database schema, connections, and migrations
│   ├── employee.py            # Employee CRUD and dataset folder management
│   ├── attendance.py          # Check-in, check-out, duration & KPI logic
│   ├── models.py              # Pydantic schemas for REST API
│   └── main.py                # FastAPI REST server
│
├── frontend/
│   └── dashboard.py           # Streamlit Web Portal (Dashboard, Kiosk, Admin)
│
├── recognition/
│   ├── register_employee.py   # Webcam & image face capture pipeline
│   ├── train_model.py         # LBPH face recognizer training engine
│   └── recognize_employee.py  # Real-time OpenCV live camera kiosk
│
├── database/
│   └── attendance.db          # SQLite relational database
│
├── employee_images/           # Face image datasets organized by employee ID_Name
│   ├── 101_mango/
│   ├── 102_chiku/
│   └── 103_sukanya/
│
├── trainer/
│   ├── trainer.yml            # Trained LBPH model weights
│   └── labels.json            # Class label-to-employee mapping
│
├── haarcascade/
│   └── haarcascade_frontalface_default.xml # OpenCV Haar Cascade detector
│
├── run.py                     # Unified CLI launcher
├── run_dashboard.bat          # 1-Click launcher for Streamlit Web Portal
├── run_kiosk.bat              # 1-Click launcher for Live Face Recognition
├── run_api.bat                # 1-Click launcher for FastAPI REST Server
├── check_db.py                # Database inspector utility
├── reset_database.py          # Database cleanup utility
├── requirements.txt           # Project dependencies
└── README.md                  # Comprehensive documentation
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- **Python 3.10, 3.11, 3.12, or 3.14**
- A working webcam (built-in or USB)

### 2. Install Dependencies
Open PowerShell or Command Prompt in the project directory:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run the Project

You can run the project using any of the convenient methods below:

### Method 1: The Unified CLI Launcher (Recommended)

Simply run:
```bash
python run.py
```
This opens an interactive menu where you can choose:
1. 📊 Launch Web Portal Dashboard (Streamlit)
2. 📸 Launch Live Face Recognition Kiosk (OpenCV)
3. 👤 Register New Employee & Capture Face
4. 🧠 Train / Retrain AI Face Recognition Model
5. 🌐 Launch FastAPI REST Server (Swagger Docs)
6. 🗄️ Inspect SQLite Database

Or use direct flags:
- `python run.py --dashboard` (Web Portal)
- `python run.py --kiosk` (OpenCV Kiosk)
- `python run.py --register` (Employee Registration)
- `python run.py --train` (Model Training)
- `python run.py --api` (FastAPI REST Server)

---

### Method 2: Double-Click Batch Files (Windows)

- **`run_dashboard.bat`**: Double-click to start the Streamlit Web Portal.
- **`run_kiosk.bat`**: Double-click to start the real-time OpenCV camera kiosk.
- **`run_api.bat`**: Double-click to launch the FastAPI backend.

---

### Method 3: Direct Command Line Execution

#### 1. Launch the Web Dashboard
```bash
streamlit run frontend/dashboard.py
```
Your default browser will automatically open: `http://localhost:8501`

#### 2. Launch the Desktop Live Face Kiosk
```bash
python -m recognition.recognize_employee
```
* **Keyboard Hotkeys**:
  - `m`: Switch between **Check-In** and **Check-Out** mode.
  - `c`: Clear cooldown cache.
  - `q` or `ESC`: Quit kiosk.

#### 3. Register a New Employee (CLI)
```bash
python -m recognition.register_employee
```
Follow the terminal prompts for Employee ID, Name, and Department, then look at the camera to collect 50 face frames.

#### 4. Train the Face Recognition Model
```bash
python -m recognition.train_model
```

#### 5. Launch the FastAPI Backend
```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```
Open interactive Swagger API documentation at: `http://127.0.0.1:8000/docs`

---

## 📊 Database Schema

### `employees` Table
| Column | Type | Description |
|---|---|---|
| `employee_id` | TEXT (PK) | Unique employee identifier |
| `employee_name` | TEXT | Full name |
| `department` | TEXT | Department / Team |
| `email` | TEXT | Optional work email |
| `created_at` | TEXT | Registration timestamp |

### `attendance` Table
| Column | Type | Description |
|---|---|---|
| `id` | INTEGER (PK AUTO) | Unique record ID |
| `employee_id` | TEXT | Linked employee ID |
| `employee_name` | TEXT | Employee name |
| `date` | TEXT | Date (`YYYY-MM-DD`) |
| `check_in` | TEXT | Check-in timestamp (`HH:MM:SS`) |
| `check_out` | TEXT | Check-out timestamp (`HH:MM:SS`) |
| `duration` | TEXT | Total hours & minutes worked (`Xh Ym Zs`) |
| `status` | TEXT | `Present`, `Present (Late)`, etc. |
| `notes` | TEXT | Notes, tags, or late arrival comments |

---

## 🚀 Cloud Deployment on Render (Render.com)

This repository includes full configuration for 1-click cloud deployment on [Render](https://render.com).

### Quick Deployment Steps:

1. **Push your code to GitHub**:
   ```bash
   git add .
   git commit -m "Configure project for Render deployment"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```

2. **Deploy on Render**:
   - Log into [Render.com](https://dashboard.render.com).
   - Click **New +** $\rightarrow$ **Web Service**.
   - Connect your GitHub repository.
   - Fill in the service configuration:
     - **Name**: `ai-attendance-portal`
     - **Runtime**: `Python`
     - **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
     - **Start Command**: `streamlit run frontend/dashboard.py --server.port $PORT --server.address 0.0.0.0`
     - **Plan**: `Free`
   - Click **Deploy Web Service**!

3. **Access Your Live App**:
   - Render automatically provisions free HTTPS (`https://<your-subdomain>.onrender.com`).
   - Open your URL to use the interactive dashboard and in-browser camera face attendance!

---

## 💡 Tips & Best Practices
- **Lighting**: For highest face recognition accuracy, ensure good front-facing lighting without strong backlighting.
- **Retraining**: Whenever you add a new employee or new face photos, click **"Train / Retrain Model Now"** in the **AI Model Trainer** tab (or run `python -m recognition.train_model`).
- **Confidence Threshold**: You can adjust the LBPH confidence distance in the **Kiosk** or **Settings** tab. Default is `70` (lower = stricter match, higher = more lenient).
