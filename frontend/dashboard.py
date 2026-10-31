"""
AI Attendance Management System - Modern Executive Portal
Streamlit 2.0 Web Application featuring:
- Executive Analytics Dashboard with Interactive Plotly Visualizations
- Browser-based Face Recognition & 1-Click Desktop OpenCV Kiosk Launcher
- Employee Directory & Multi-method Registration (Webcam Snapshot + Photo Upload)
- Comprehensive Attendance Logs with Working Hours & Manual Overrides
- AI Model Training Center & Dataset Explorer
- System Diagnostics, Settings, and Database Backup
"""
import sys
import io
import os
import json
import subprocess
from pathlib import Path
from datetime import datetime, date, timedelta
import calendar
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import textwrap
from PIL import Image

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import importlib
import backend.config as backend_cfg
import backend.database as backend_db
import backend.employee as backend_emp
import backend.attendance as backend_att
import backend.payroll as backend_pay

importlib.reload(backend_cfg)
importlib.reload(backend_db)
importlib.reload(backend_emp)
importlib.reload(backend_att)
importlib.reload(backend_pay)

from backend.config import (
    DB_PATH, HAAR_CASCADE_PATH, TRAINER_MODEL_PATH, LABELS_PATH,
    DATASET_DIR, DEFAULT_CONFIDENCE_THRESHOLD,
    WORK_START_TIME, CHECKIN_GRACE_TIME, WORK_END_TIME, CHECKOUT_GRACE_TIME,
    DAILY_WORK_HOURS, DEFAULT_HOURLY_RATE, DEFAULT_DAILY_SALARY, HALF_DAY_FACTOR, CURRENCY_SYMBOL,
    DEFAULT_CAMERA_INDEX, FACE_IMAGE_SIZE
)
from backend.database import get_connection, init_db, calculate_duration
from backend.employee import (
    add_employee, get_all_employees, get_employee,
    delete_employee, update_employee, get_departments
)
from backend.attendance import (
    mark_attendance, record_manual_attendance,
    delete_attendance_record, get_today_summary
)
from backend.payroll import (
    calculate_all_payroll, calculate_employee_payroll, save_finalized_payroll
)
from recognition.train_model import train_model
from recognition.register_employee import process_and_save_face_images, capture_employee_via_webcam
from recognition.recognize_employee import predict_single_frame, play_chime

# Ensure database is up to date
init_db()

# Page configuration
st.set_page_config(
    page_title="AI Attendance Portal",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Global styles */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1E88E5, #7E57C2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #888888;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.02));
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.1);
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #888;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-present { background-color: #2e7d32; color: #ffffff; }
    .badge-absent { background-color: #c62828; color: #ffffff; }
    .badge-checkout { background-color: #0277bd; color: #ffffff; }
    .badge-late { background-color: #ef6c00; color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("## 📷 AI Attendance 2.0")
    st.caption("Computer Vision & Facial Biometrics")
    st.divider()

    menu = st.radio(
        "Navigation",
        [
            "📊 Executive Dashboard",
            "📷 Face Recognition Kiosk",
            "👥 Employee Management",
            "📋 Attendance Logs",
            "💼 Payroll & Compensation",
            "🧠 AI Model Trainer",
            "⚙️ System Settings"
        ],
        index=0
    )

    st.divider()

    # System Health Indicator
    st.markdown("### 🖥️ System Status")
    is_model_trained = TRAINER_MODEL_PATH.exists() and LABELS_PATH.exists()
    labels_count = 0
    if is_model_trained:
        try:
            with open(LABELS_PATH, "r") as f:
                labels_count = len(json.load(f))
        except Exception:
            pass
    is_model_trained = is_model_trained and (labels_count > 0)

    st.write(f"🗄️ **Database:** Connected")
    if is_model_trained:
        st.write(f"🧠 **AI Model:** Trained ({labels_count} employees)")
    else:
        st.warning("🧠 **AI Model:** Not Trained")
    st.write(f"📸 **Detector:** Haar Cascade Ready")

    st.divider()
    today_str = datetime.now().strftime("%A, %d %B %Y")
    current_time_str = datetime.now().strftime("%H:%M:%S")
    st.caption(f"📅 {today_str}\n🕒 System Time: {current_time_str}")

# ----------------- PAGE 1: EXECUTIVE DASHBOARD -----------------
if menu == "📊 Executive Dashboard":
    st.markdown('<div class="main-title">📊 Attendance Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Real-time attendance tracking, department summaries, and workforce metrics</div>', unsafe_allow_html=True)

    # Date filter controls
    col_filter1, col_filter2 = st.columns([2, 3])
    with col_filter1:
        date_preset = st.selectbox(
            "Filter Period",
            ["Today", "Yesterday", "Last 7 Days", "Last 30 Days", "Custom Range", "All Time"],
            index=0
        )

    today = date.today()
    if date_preset == "Today":
        start_date = today
        end_date = today
    elif date_preset == "Yesterday":
        start_date = today - timedelta(days=1)
        end_date = today - timedelta(days=1)
    elif date_preset == "Last 7 Days":
        start_date = today - timedelta(days=7)
        end_date = today
    elif date_preset == "Last 30 Days":
        start_date = today - timedelta(days=30)
        end_date = today
    elif date_preset == "All Time":
        start_date = date(2020, 1, 1)
        end_date = today + timedelta(days=1)
    else:
        with col_filter2:
            custom_dates = st.date_input("Select Date Range", [today - timedelta(days=7), today])
            if isinstance(custom_dates, (list, tuple)) and len(custom_dates) == 2:
                start_date, end_date = custom_dates
            else:
                start_date = today
                end_date = today

    # Query attendance & employees
    conn = get_connection()
    employees_df = pd.read_sql_query("SELECT * FROM employees", conn)
    att_query = """
        SELECT * FROM attendance
        WHERE date BETWEEN ? AND ?
        ORDER BY date DESC, id DESC
    """
    attendance_df = pd.read_sql_query(att_query, conn, params=(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")))
    conn.close()

    total_employees = len(employees_df)

    # Calculate metrics for the selected range
    if date_preset in ["Today", "Yesterday"]:
        present_count = attendance_df["employee_id"].nunique() if not attendance_df.empty else 0
        checked_out_count = len(attendance_df[attendance_df["check_out"].str.strip() != ""]) if not attendance_df.empty else 0
        currently_in = max(present_count - checked_out_count, 0)
        absent_count = max(total_employees - present_count, 0)
        attendance_pct = (present_count / total_employees * 100) if total_employees > 0 else 0.0
    else:
        # Multi-day metrics
        total_records = len(attendance_df)
        present_count = total_records
        checked_out_count = len(attendance_df[attendance_df["check_out"].str.strip() != ""]) if not attendance_df.empty else 0
        currently_in = max(present_count - checked_out_count, 0)
        unique_attendees = attendance_df["employee_id"].nunique() if not attendance_df.empty else 0
        absent_count = max(total_employees - unique_attendees, 0)
        attendance_pct = (unique_attendees / total_employees * 100) if total_employees > 0 else 0.0

    # Display KPI Cards
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">👥 Total Workforce</div>
            <div class="metric-value" style="color: #42A5F5;">{total_employees}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">✅ Present</div>
            <div class="metric-value" style="color: #66BB6A;">{present_count}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🏢 Currently In Office</div>
            <div class="metric-value" style="color: #FFA726;">{currently_in}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🚪 Checked Out</div>
            <div class="metric-value" style="color: #29B6F6;">{checked_out_count}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📈 Attendance Rate</div>
            <div class="metric-value" style="color: #AB47BC;">{attendance_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Visual Charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("📊 Status Breakdown")
        if not attendance_df.empty or total_employees > 0:
            pie_data = pd.DataFrame({
                "Status": ["Present & Active", "Checked Out", "Absent"],
                "Count": [currently_in, checked_out_count, absent_count]
            })
            # Filter zero entries
            pie_data = pie_data[pie_data["Count"] > 0]
            if not pie_data.empty:
                fig_donut = px.pie(
                    pie_data,
                    names="Status",
                    values="Count",
                    hole=0.55,
                    color="Status",
                    color_discrete_map={
                        "Present & Active": "#66BB6A",
                        "Checked Out": "#29B6F6",
                        "Absent": "#EF5350"
                    }
                )
                fig_donut.update_traces(textposition='inside', textinfo='percent+label')
                fig_donut.update_layout(
                    showlegend=True,
                    margin=dict(t=20, b=20, l=20, r=20),
                    height=320
                )
                st.plotly_chart(fig_donut, use_container_width=True)
            else:
                st.info("No attendance activity recorded for this period.")
        else:
            st.info("No employee records found in system.")

    with chart_col2:
        st.subheader("🏢 Department Attendance")
        if not employees_df.empty:
            merged_att = attendance_df.merge(
                employees_df[["employee_id", "department"]],
                on="employee_id",
                how="left",
                suffixes=("", "_emp")
            )
            dept_counts = employees_df.groupby("department").size().reset_index(name="Total")

            if not merged_att.empty:
                dept_col = "department_emp" if "department_emp" in merged_att.columns else "department"
                dept_present = merged_att.groupby(dept_col).size().reset_index(name="Present")
                dept_summary = dept_counts.merge(
                    dept_present,
                    left_on="department",
                    right_on=dept_col,
                    how="left"
                ).fillna(0)
            else:
                dept_summary = dept_counts
                dept_summary["Present"] = 0

            dept_summary["Absent"] = dept_summary["Total"] - dept_summary["Present"]
            dept_summary["Absent"] = dept_summary["Absent"].apply(lambda x: max(x, 0))

            fig_dept = go.Figure()
            fig_dept.add_trace(go.Bar(
                name="Present",
                x=dept_summary["department"],
                y=dept_summary["Present"],
                marker_color="#66BB6A"
            ))
            fig_dept.add_trace(go.Bar(
                name="Absent",
                x=dept_summary["department"],
                y=dept_summary["Absent"],
                marker_color="#EF5350"
            ))
            fig_dept.update_layout(
                barmode="stack",
                margin=dict(t=20, b=20, l=20, r=20),
                height=320,
                xaxis_title="Department",
                yaxis_title="Employees"
            )
            st.plotly_chart(fig_dept, use_container_width=True)
        else:
            st.info("No departments registered.")

    # Attendance Trend Chart (if multi-day selected)
    if date_preset in ["Last 7 Days", "Last 30 Days", "Custom Range", "All Time"] and not attendance_df.empty:
        st.subheader("📈 Daily Attendance Trend")
        trend_df = attendance_df.groupby("date").size().reset_index(name="Present Count")
        trend_df = trend_df.sort_values("date")
        fig_trend = px.area(
            trend_df,
            x="date",
            y="Present Count",
            markers=True,
            title="Attendance Over Time",
            color_discrete_sequence=["#42A5F5"]
        )
        fig_trend.update_layout(height=280, margin=dict(t=30, b=20, l=20, r=20))
        st.plotly_chart(fig_trend, use_container_width=True)



# ----------------- PAGE 2: FACE RECOGNITION KIOSK -----------------
elif menu == "📷 Face Recognition Kiosk":
    st.markdown('<div class="main-title">📷 Face Recognition Attendance Kiosk</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Mark attendance via web camera snapshot or high-speed desktop live kiosk</div>', unsafe_allow_html=True)

    if not is_model_trained:
        st.error("⚠️ AI Face Recognition Model is not trained yet. Please register employees and train the model in the 'AI Model Trainer' tab first.")
    else:
        mode_col1, mode_col2 = st.columns([1, 1])
        with mode_col1:
            kiosk_mode = st.radio("Select Kiosk Action", ["🟢 Check-In", "🟠 Check-Out"], horizontal=True)
            mode_arg = "checkin" if "Check-In" in kiosk_mode else "checkout"

        with mode_col2:
            min_confidence_val = st.slider(
                "Minimum Match Confidence Required (%)",
                min_value=50,
                max_value=90,
                value=60,
                help="Higher threshold prevents unknown/unregistered faces from matching. Recommended: 60% - 65%."
            )
            # Convert percentage to max allowed LBPH distance (e.g., 60% confidence -> distance <= 40)
            threshold_val = 100 - min_confidence_val

        st.divider()

        kiosk_tabs = st.tabs(["🌐 Web Camera Instant Scan", "🖥️ High-Speed Desktop Live Kiosk"])

        # Tab 1: In-Browser Web Camera Snapshot
        with kiosk_tabs[0]:
            st.write("### 📸 Snapshot Recognition")
            st.caption("Look straight at your webcam and take a snapshot to instantly mark attendance.")

            cam_image = st.camera_input("Smile at the camera to verify face ID")

            if cam_image is not None:
                try:
                    file_bytes = np.asarray(bytearray(cam_image.read()), dtype=np.uint8)
                    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                    with st.spinner("Analyzing face biometrics..."):
                        try:
                            results = predict_single_frame(
                                frame,
                                confidence_threshold=threshold_val,
                                min_confidence_percent=min_confidence_val
                            )
                        except TypeError:
                            # Resilient fallback if Streamlit has the older module cached in memory
                            results = predict_single_frame(frame, confidence_threshold=threshold_val)
                            for item in results:
                                if item.get("confidence_percent", 0) < min_confidence_val:
                                    item["is_recognized"] = False
                                    item["raw_name"] = "Unknown"
                                    item["employee_name"] = "Unknown"

                    if not results:
                        st.warning("⚠️ No face detected in frame. Please ensure your face is well-lit and facing the camera.")
                    else:
                        for item in results:
                            if item["is_recognized"]:
                                play_chime(1400, 150)
                                att_res = mark_attendance(
                                    item["raw_name"],
                                    mode=mode_arg
                                )

                                (x, y, w, h) = item["box"]
                                face_crop = frame[max(0, y):y+h, max(0, x):x+w]
                                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

                                res_col1, res_col2 = st.columns([1, 3])
                                with res_col1:
                                    st.image(face_rgb, caption=f"{item['employee_name']}", width=160)
                                with res_col2:
                                    status_code = att_res.get("status")
                                    if status_code == "checkin_success":
                                        status_tag = att_res.get("attendance_status", "Present")
                                        if status_tag == "Half Day":
                                            st.warning(f"⚠️ **Check-In Recorded as Half Day**\n\n**Welcome, {att_res['employee_name']}** (ID: {att_res['employee_id']})\n- Timestamp: {att_res['time']} (After 09:30 AM)\n- Day Status: **Half Day (Late Arrival)**\n- Wage: **₹100.00 / hr** (Computed at check-out based on exact hours)\n- Department: {att_res['department']}")
                                        else:
                                            st.success(f"🎉 **On-Time Morning Check-In**\n\n**Welcome, {att_res['employee_name']}** (ID: {att_res['employee_id']})\n- Timestamp: {att_res['time']} (09:00 - 09:30 AM)\n- Day Status: **On-Time (Present)**\n- Wage: **₹100.00 / hr** (Computed at check-out based on exact hours)\n- Department: {att_res['department']}")
                                    elif status_code == "already_checked_in":
                                        st.info(f"ℹ️ **Already Checked In Today**\n\n**{att_res['employee_name']}** (ID: {att_res['employee_id']})\n- Check-In Time: {att_res['time']}")
                                    elif status_code == "checkout_success":
                                        st.success(f"👋 **Check-Out Successful!**\n\n**Goodbye, {att_res['employee_name']}** (ID: {att_res['employee_id']})\n- Check-Out Time: {att_res['time']}\n- Total Working Duration: **{att_res.get('duration', 'N/A')}**\n- Total Hours: **{att_res.get('hours_worked', 0):.2f} hrs**\n- Salary Earned Today: **₹{att_res.get('salary_earned', 0):.2f}** (at ₹100/hr)")
                                    elif status_code == "already_checked_out":
                                        st.info(f"ℹ️ **Already Checked Out**\n\n**{att_res['employee_name']}** checked out earlier today at {att_res['time']}.")
                                    elif status_code == "not_checked_in":
                                        st.error(f"❌ **Cannot Check-Out!**\n\n**{att_res['employee_name']}** has not checked in today yet.")
                                    st.caption(f"Match Confidence: {item['confidence_percent']}% | LBPH Distance: {item['confidence_distance']} (Threshold: ≥{min_confidence_val}%)")
                            else:
                                (x, y, w, h) = item["box"]
                                face_crop = frame[max(0, y):y+h, max(0, x):x+w]
                                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                                res_col1, res_col2 = st.columns([1, 3])
                                with res_col1:
                                    st.image(face_rgb, caption="Unregistered Person", width=140)
                                with res_col2:
                                    st.error(
                                        f"❌ **Employee Not Registered!**\n\n"
                                        f"This face does not match any registered employee in the system.\n\n"
                                        f"- **Match Confidence:** {item['confidence_percent']}% (Required minimum: **{min_confidence_val}%**)\n"
                                        f"- **Status:** Access Denied / Not Registered\n\n"
                                        f"Please register this person in the **Employee Management** tab first."
                                    )
                                    st.caption(f"LBPH Distance: {item['confidence_distance']} | Confidence: {item['confidence_percent']}%")
                except Exception as e:
                    st.error(f"Error during recognition: {e}")

        # Tab 2: Desktop OpenCV Live Kiosk Launcher
        with kiosk_tabs[1]:
            st.write("### 🖥️ Dedicated Live OpenCV Kiosk")
            st.info("Launches a 30-FPS desktop window with real-time face tracking, audio feedback, and keyboard controls.")
            st.markdown("""
            **Controls inside Live Kiosk:**
            - Press **`m`** on keyboard to toggle between **Check-In** and **Check-Out**.
            - Press **`c`** to clear cooldown cache.
            - Press **`q`** or **`ESC`** to exit kiosk window.
            """)

            if st.button("🚀 Launch Real-Time Desktop Face Kiosk", type="primary"):
                # Detect headless cloud environment (Render, Linux server without X11)
                is_cloud_headless = bool(os.environ.get("RENDER") or (sys.platform != "win32" and not os.environ.get("DISPLAY")))
                if is_cloud_headless:
                    st.warning("⚠️ The Desktop OpenCV Kiosk requires a physical monitor and local display directly connected to the host machine. On cloud hosting (Render), please use **Tab 1: In-Browser Web Camera** to mark attendance directly using your web browser.")
                else:
                    try:
                        # Launch recognition module in background process
                        cmd = [sys.executable, "-m", "recognition.recognize_employee"]
                        subprocess.Popen(cmd, cwd=str(BASE_DIR))
                        st.success("✅ Live Face Kiosk launched in a new desktop window! Look for the camera window.")
                    except Exception as e:
                        st.error(f"Failed to launch kiosk: {e}")

    st.divider()
    # Today's Recent Scans
    st.subheader("🕒 Today's Live Attendance Feed")
    today_summary = get_today_summary()
    recent_records = today_summary.get("records", [])
    if recent_records:
        recent_df = pd.DataFrame(recent_records).sort_values("id", ascending=False)
        st.dataframe(
            recent_df[["employee_id", "employee_name", "check_in", "check_out", "duration", "status", "notes"]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No attendance recorded yet today.")

# ----------------- PAGE 3: EMPLOYEE MANAGEMENT -----------------
elif menu == "👥 Employee Management":
    st.markdown('<div class="main-title">👥 Employee Directory & Registration</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Manage employee profiles and collect face biometric datasets</div>', unsafe_allow_html=True)

    emp_tabs = st.tabs(["📋 Employee Directory", "➕ Register New Employee", "✏️ Edit / Delete Employee"])

    # Tab 1: Directory
    with emp_tabs[0]:
        employees = get_all_employees()
        if employees:
            emp_df = pd.DataFrame(employees)
            col_search, col_dept = st.columns([2, 1])
            with col_search:
                search_q = st.text_input("Search employees", placeholder="Search by name, ID, or email...")
            with col_dept:
                depts = ["All"] + get_departments()
                selected_dept = st.selectbox("Filter Department", depts)

            filtered_emp = emp_df
            if search_q:
                filtered_emp = filtered_emp[
                    filtered_emp["employee_name"].str.contains(search_q, case=False, na=False) |
                    filtered_emp["employee_id"].str.contains(search_q, case=False, na=False) |
                    filtered_emp["email"].str.contains(search_q, case=False, na=False)
                ]
            if selected_dept != "All":
                filtered_emp = filtered_emp[filtered_emp["department"] == selected_dept]

            st.write(f"Total Employees: **{len(filtered_emp)}**")

            # Display cards or table
            for _, row in filtered_emp.iterrows():
                with st.expander(f"👤 {row['employee_name']} (ID: {row['employee_id']}) - {row['department']}"):
                    col1, col2 = st.columns([1, 3])
                    # Check if face sample exists
                    emp_folder = DATASET_DIR / f"{row['employee_id']}_{row['employee_name']}"
                    sample_img = None
                    if emp_folder.exists():
                        img_files = list(emp_folder.glob("*.jpg"))
                        if img_files:
                            sample_img = img_files[0]

                    with col1:
                        if sample_img:
                            st.image(str(sample_img), caption="Biometric Sample", width=120)
                        else:
                            st.caption("No face images captured.")
                    with col2:
                        st.write(f"**Employee ID:** `{row['employee_id']}`")
                        st.write(f"**Department:** {row['department']}")
                        st.write(f"**Email:** {row['email'] or 'Not provided'}")
                        st.write(f"**Hourly Wage Rate:** ₹{float(row.get('hourly_rate') or 100):.0f} / hour")
                        st.write(f"**Standard 8h Shift:** ₹{float(row.get('daily_salary') or 800):.0f} / day")
                        st.write(f"**Registered At:** {row.get('created_at', 'N/A')}")
        else:
            st.info("No employees registered yet. Go to 'Register New Employee' tab to add your first employee!")

    # Tab 2: Register New Employee
    with emp_tabs[1]:
        st.write("### ➕ Register New Employee & Face Biometrics")
        with st.form("register_form"):
            reg_col1, reg_col2 = st.columns(2)
            with reg_col1:
                new_id = st.text_input("Employee ID *", placeholder="e.g. 104")
                new_name = st.text_input("Full Name *", placeholder="e.g. Sarah Connor")
                new_hourly = st.number_input("Hourly Wage Rate (Rs/hr) *", min_value=10.0, max_value=5000.0, value=100.0, step=10.0,
                                             help="Salary is calculated as: Total hours available in company (check-in to check-out) * Hourly Rate. Default is Rs 100/hr.")
            with reg_col2:
                new_dept = st.text_input("Department *", placeholder="e.g. Operations")
                new_email = st.text_input("Work Email", placeholder="e.g. sarah@company.com")

            st.markdown("#### 📸 Face Image Samples")
            st.caption("Upload multiple face photos (recommended: 5-20 images with various expressions and angles).")
            uploaded_photos = st.file_uploader(
                "Upload Face Photos (.jpg, .jpeg, .png)",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True
            )

            submit_reg = st.form_submit_button("✅ Register Employee & Save Biometrics", type="primary")

            if submit_reg:
                if not new_id.strip() or not new_name.strip() or not new_dept.strip():
                    st.error("Please fill in all required fields (ID, Name, Department).")
                else:
                    success, msg = add_employee(new_id, new_name, new_dept, new_email, hourly_rate=new_hourly, daily_salary=new_hourly * 8.0)
                    if not success:
                        st.error(msg)
                    else:
                        st.success(msg)
                        if uploaded_photos:
                            saved_count, p_msg = process_and_save_face_images(new_id, new_name, uploaded_photos)
                            st.info(p_msg)
                        else:
                            st.warning("Note: No face photos were uploaded. You can capture face samples using webcam below.")

        st.divider()
        st.write("#### 🎥 Capture Face Samples via Webcam")
        st.caption("Capture photos in real-time if photos were not uploaded above.")
        cam_snap = st.camera_input("Take photo of employee")
        if cam_snap is not None and new_id.strip() and new_name.strip():
            if st.button("💾 Save Snapshot as Biometric Sample"):
                saved_count, p_msg = process_and_save_face_images(new_id, new_name, [cam_snap])
                st.success(f"{p_msg} Don't forget to Retrain Model in 'AI Model Trainer'!")

    # Tab 3: Edit / Delete Employee
    with emp_tabs[2]:
        st.write("### ✏️ Edit or Remove Employee")
        all_emps = get_all_employees()
        if all_emps:
            emp_options = {f"{e['employee_id']} - {e['employee_name']}": e for e in all_emps}
            chosen_key = st.selectbox("Select Employee to Manage", list(emp_options.keys()))
            selected_emp = emp_options[chosen_key]

            edit_col1, edit_col2 = st.columns(2)
            with edit_col1:
                st.markdown("#### Update Details")
                with st.form("edit_form"):
                    updated_name = st.text_input("Full Name", value=selected_emp["employee_name"])
                    updated_dept = st.text_input("Department", value=selected_emp["department"])
                    updated_email = st.text_input("Email", value=selected_emp["email"] or "")
                    updated_hourly = st.number_input("Hourly Wage Rate (Rs/hr)", min_value=10.0, max_value=5000.0,
                                                     value=float(selected_emp.get("hourly_rate") or 100.0), step=10.0,
                                                     help="Standard wage rate is Rs 100/hr.")
                    save_edit = st.form_submit_button("Update Details")
                    if save_edit:
                        u_ok, u_msg = update_employee(selected_emp["employee_id"], updated_name, updated_dept, updated_email,
                                                      hourly_rate=updated_hourly, daily_salary=updated_hourly * 8.0)
                        if u_ok:
                            st.success(u_msg)
                            st.rerun()
                        else:
                            st.error(u_msg)

            with edit_col2:
                st.markdown("#### Danger Zone")
                st.warning(f"Delete employee **{selected_emp['employee_name']}** (ID: {selected_emp['employee_id']})?")
                del_attendance = st.checkbox("Also delete all attendance records for this employee", value=False)
                if st.button("🗑️ Delete Employee Permanently", type="secondary"):
                    d_ok, d_msg = delete_employee(selected_emp["employee_id"], delete_images=True, delete_attendance=del_attendance)
                    if d_ok:
                        st.success(d_msg)
                        st.rerun()
                    else:
                        st.error(d_msg)
        else:
            st.info("No employees found.")

# ----------------- PAGE 4: ATTENDANCE LOGS -----------------
elif menu == "📋 Attendance Logs":
    st.markdown('<div class="main-title">📋 Attendance Logs & Records</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Filter historical logs, calculate working hours, or add manual attendance overrides</div>', unsafe_allow_html=True)

    log_tabs = st.tabs(["📜 Historical Logs", "✍️ Manual Attendance Entry", "🗑️ Manage Records"])

    # Tab 1: Historical Logs
    with log_tabs[0]:
        conn = get_connection()
        query = "SELECT * FROM attendance ORDER BY date DESC, id DESC"
        full_att_df = pd.read_sql_query(query, conn)
        conn.close()

        if not full_att_df.empty:
            f1, f2, f3 = st.columns(3)
            with f1:
                all_dates = ["All Dates"] + sorted(full_att_df["date"].unique().tolist(), reverse=True)
                sel_date = st.selectbox("Filter Date", all_dates)
            with f2:
                all_status = ["All Statuses"] + sorted(full_att_df["status"].unique().tolist())
                sel_status = st.selectbox("Filter Status", all_status)
            with f3:
                search_kw = st.text_input("Search Employee Name / ID", "")

            filtered = full_att_df
            if sel_date != "All Dates":
                filtered = filtered[filtered["date"] == sel_date]
            if sel_status != "All Statuses":
                filtered = filtered[filtered["status"] == sel_status]
            if search_kw:
                filtered = filtered[
                    filtered["employee_name"].str.contains(search_kw, case=False, na=False) |
                    filtered["employee_id"].astype(str).str.contains(search_kw, case=False, na=False)
                ]

            st.write(f"Showing **{len(filtered)}** record(s)")
            st.dataframe(filtered, use_container_width=True, hide_index=True)

            # Export options
            csv_exp = filtered.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Filtered Log (CSV)", csv_exp, "attendance_logs.csv", "text/csv")
        else:
            st.info("No attendance records in the database.")

    # Tab 2: Manual Attendance Entry
    with log_tabs[1]:
        st.write("### ✍️ Add / Override Attendance Record")
        st.caption("Use this form if an employee was unable to scan their face or for HR manual adjustments.")

        emps = get_all_employees()
        if emps:
            with st.form("manual_attendance_form"):
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    emp_choices = {f"{e['employee_id']} - {e['employee_name']}": e['employee_id'] for e in emps}
                    m_emp = st.selectbox("Employee", list(emp_choices.keys()))
                    m_date = st.date_input("Date", date.today())
                    m_status = st.selectbox("Status", ["Present", "Half Day", "On Leave", "Excused"])
                with m_col2:
                    m_checkin = st.text_input("Check-In Time (HH:MM:SS)", value="09:00:00")
                    m_checkout = st.text_input("Check-Out Time (HH:MM:SS)", value="17:00:00")
                    m_notes = st.text_input("Reason / Notes", value="Manual HR Entry")

                submit_manual = st.form_submit_button("💾 Save Attendance Record", type="primary")
                if submit_manual:
                    target_id = emp_choices[m_emp]
                    ok, msg = record_manual_attendance(
                        target_id,
                        m_date.strftime("%Y-%m-%d"),
                        m_checkin.strip(),
                        m_checkout.strip(),
                        status=m_status,
                        notes=m_notes.strip()
                    )
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
        else:
            st.warning("Please register employees first.")

    # Tab 3: Delete Record
    with log_tabs[2]:
        st.write("### 🗑️ Delete Specific Attendance Record")
        conn = get_connection()
        records_df = pd.read_sql_query("SELECT id, employee_id, employee_name, date, check_in, check_out FROM attendance ORDER BY id DESC LIMIT 50", conn)
        conn.close()

        if not records_df.empty:
            del_id = st.number_input("Enter Record ID to Delete", min_value=1, step=1)
            if st.button("Delete Record", type="secondary"):
                delete_attendance_record(del_id)
                st.success(f"Record #{del_id} deleted successfully.")
                st.rerun()
            st.caption("Recent records for reference:")
            st.dataframe(records_df, use_container_width=True, hide_index=True)
        else:
            st.info("No records to manage.")

# ----------------- PAGE 5: PAYROLL & COMPENSATION -----------------
elif menu == "💼 Payroll & Compensation":
    st.markdown('<div class="main-title">💼 Employee Payroll & Compensation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Automated salary computation based on 8-hour shift, punctuality, and attendance</div>', unsafe_allow_html=True)

    # Shift & Policy Banner Card
    st.markdown("""
    <div style="background: rgba(30, 136, 229, 0.08); border-left: 4px solid #1E88E5; padding: 14px 20px; border-radius: 8px; margin-bottom: 22px;">
        <h4 style="margin: 0 0 8px 0; color: #1E88E5;">⏰ Company Work Shift & Salary Rules</h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; font-size: 0.95rem;">
            <div>• <strong>Hourly Wage Rate:</strong> ₹100.00 / hour</div>
            <div>• <strong>Daily Net Salary:</strong> Hours Worked × ₹100 / day</div>
            <div>• <strong>Monthly Net Salary:</strong> Total of all days' net salary</div>
            <div>• <strong>Standard Shift:</strong> 9:00 AM – 5:00 PM (8 Hours = ₹800.00)</div>
            <div>• <strong>Check-In Window:</strong> 9:00 AM – 9:30 AM (<strong>On-Time Shift</strong>)</div>
            <div>• <strong>Late Check-In:</strong> After 9:30 AM (<strong>Marked Half Day</strong>)</div>
            <div>• <strong>Check-Out Window:</strong> 5:00 PM – 5:30 PM (8 hrs completed)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Month / Period Filters - All 12 Months + Custom Range
    all_months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
        "Custom Date Range"
    ]
    today_d = date.today()
    default_month_idx = today_d.month - 1

    col_p1, col_p2, col_p3 = st.columns([2.2, 1.2, 2.6])
    with col_p1:
        pay_period_choice = st.selectbox(
            "Select Payroll Month",
            all_months,
            index=default_month_idx,
            help="Select any of the 12 calendar months, or select 'Custom Date Range' for custom dates."
        )

    with col_p2:
        current_year = today_d.year
        year_choices = [current_year, current_year - 1, current_year - 2, current_year + 1]
        selected_year = st.selectbox("Year", year_choices, index=0)

    if pay_period_choice == "Custom Date Range":
        with col_p3:
            custom_p = st.date_input(
                "Select Custom Date Range",
                [date(selected_year, today_d.month, 1), today_d],
                help="Pick arbitrary start and end dates."
            )
            if isinstance(custom_p, (list, tuple)) and len(custom_p) == 2:
                p_start, p_end = custom_p
            elif isinstance(custom_p, (list, tuple)) and len(custom_p) == 1:
                p_start, p_end = custom_p[0], custom_p[0]
            else:
                p_start, p_end = today_d, today_d
    else:
        month_num = all_months.index(pay_period_choice) + 1
        num_days = calendar.monthrange(selected_year, month_num)[1]
        p_start = date(selected_year, month_num, 1)
        p_end = date(selected_year, month_num, num_days)

        with col_p3:
            st.markdown(f"""
            <div style="background: rgba(30, 136, 229, 0.08); border-left: 3px solid #1E88E5; padding: 10px 14px; border-radius: 8px; margin-top: 24px; font-size: 0.9rem;">
                🗓️ <strong>Month:</strong> {pay_period_choice} {selected_year}<br>
                <span style="color: #666; font-size: 0.82rem;">{p_start.strftime('%d %b %Y')} &mdash; {p_end.strftime('%d %b %Y')} ({num_days} Days)</span>
            </div>
            """, unsafe_allow_html=True)

    # Force reload backend payroll module so computations are always fresh
    importlib.reload(backend_pay)
    pay_summary = backend_pay.calculate_all_payroll(start_date=p_start, end_date=p_end)
    emp_payrolls = pay_summary.get("employee_payrolls", [])

    # Top Metric KPI Cards
    kpi_p1, kpi_p2, kpi_p3, kpi_p4, kpi_p5, kpi_p6 = st.columns(6)
    with kpi_p1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">💰 Total Payroll</div>
            <div class="metric-value" style="color: #43A047;">₹{pay_summary['total_payroll_cost']:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_p2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">👥 Employees</div>
            <div class="metric-value" style="color: #1E88E5;">{pay_summary['total_employees']}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_p3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⏱️ Total Hours</div>
            <div class="metric-value" style="color: #AB47BC;">{pay_summary['total_hours']:.2f}h</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_p4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">💵 Base Rate</div>
            <div class="metric-value" style="color: #00ACC1;">₹100/hr</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_p5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🟢 On-Time Days</div>
            <div class="metric-value" style="color: #66BB6A;">{pay_summary['total_full_days']}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_p6:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🟡 Half Days (Late)</div>
            <div class="metric-value" style="color: #FFA726;">{pay_summary['total_half_days']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Visual Analytics
    if emp_payrolls:
        ch_col1, ch_col2 = st.columns(2)
        with ch_col1:
            st.subheader("🏢 Salary Payout by Department")
            dept_pay_dict = {}
            for e in emp_payrolls:
                d = e["department"]
                dept_pay_dict[d] = dept_pay_dict.get(d, 0.0) + e["net_salary"]
            dept_df = pd.DataFrame(list(dept_pay_dict.items()), columns=["Department", "Total Payout"])
            fig_d = px.bar(dept_df, x="Department", y="Total Payout", color="Department", text_auto=".2f")
            fig_d.update_layout(height=280, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_d, use_container_width=True)

        with ch_col2:
            st.subheader("📊 Days Distribution")
            dist_df = pd.DataFrame({
                "Type": ["On-Time Shifts", "Half Days (Late Check-In)", "Absent (Unpaid)"],
                "Count": [pay_summary["total_full_days"], pay_summary["total_half_days"], pay_summary["total_absent_days"]]
            })
            dist_df = dist_df[dist_df["Count"] > 0]
            if not dist_df.empty:
                fig_dist = px.pie(
                    dist_df,
                    names="Type",
                    values="Count",
                    hole=0.55,
                    color="Type",
                    color_discrete_map={
                        "On-Time Shifts": "#66BB6A",
                        "Half Days (Late Check-In)": "#FFA726",
                        "Absent (Unpaid)": "#EF5350"
                    }
                )
                fig_dist.update_layout(height=280, margin=dict(t=20, b=20, l=20, r=20))
                st.plotly_chart(fig_dist, use_container_width=True)

    st.divider()

    # Payroll Subtabs
    pay_tabs = st.tabs(["📋 Employee Salary Sheet", "🧾 Individual Payslip Generator", "⚙️ Salary Settings"])

    # Tab 1: Salary Sheet
    with pay_tabs[0]:
        period_title = f"{pay_period_choice} {selected_year}" if pay_period_choice != "Custom Date Range" else f"{p_start.strftime('%d %b %Y')} to {p_end.strftime('%d %b %Y')}"
        st.subheader(f"📋 Salary Sheet — {period_title}")
        if emp_payrolls:
            sheet_data = []
            for e in emp_payrolls:
                sheet_data.append({
                    "Employee ID": e["employee_id"],
                    "Employee Name": e["employee_name"],
                    "Department": e["department"],
                    "Hourly Rate": f"₹{e.get('hourly_rate', 100):.0f}/hr",
                    "On-Time Days": e["full_days"],
                    "Half Days (Late)": e["half_days"],
                    "Absent Days": e["absent_days"],
                    "Time Available": e["formatted_hours"],
                    "Hours Worked": f"{e['total_hours_worked']:.2f} hrs",
                    "Monthly Net Salary": f"₹{e['net_salary']:,.2f}",
                    "Status": "Calculated"
                })
            sheet_df = pd.DataFrame(sheet_data)
            st.dataframe(sheet_df, use_container_width=True, hide_index=True)
            st.caption("ℹ️ **Calculation:** Daily Net Salary = Hours Worked per day × ₹100 | Monthly Net Salary = Total sum of all days' net salary.")

            # Export buttons
            col_e1, col_e2 = st.columns(2)
            file_slug = f"payroll_{pay_period_choice.lower().replace(' ', '_')}_{selected_year}" if pay_period_choice != "Custom Date Range" else f"payroll_{p_start}_to_{p_end}"
            with col_e1:
                csv_pay = sheet_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Salary Sheet (CSV)", csv_pay, f"{file_slug}.csv", "text/csv", use_container_width=True)
            with col_e2:
                try:
                    ex_buf = io.BytesIO()
                    with pd.ExcelWriter(ex_buf, engine="openpyxl") as wr:
                        sheet_df.to_excel(wr, index=False, sheet_name="Payroll")
                    st.download_button("📊 Download Salary Sheet (Excel)", ex_buf.getvalue(), f"{file_slug}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
                except Exception:
                    st.caption("Excel export requires openpyxl.")
        else:
            st.info("No employee records found.")

    # Tab 2: Individual Payslip Generator
    with pay_tabs[1]:
        st.subheader("🧾 Employee Pay Slip Voucher")
        if emp_payrolls:
            emp_dict_map = {f"{e['employee_id']} - {e['employee_name']}": e for e in emp_payrolls}
            selected_slip_key = st.selectbox("Select Employee for Payslip", list(emp_dict_map.keys()))
            chosen_pay = emp_dict_map[selected_slip_key]

            # Generate Payslip Document
            issue_date = datetime.now().strftime("%d %B %Y")
            slip_html = f"""
            <div style="background: #ffffff; color: #111111; padding: 30px; border-radius: 12px; border: 1px solid #e0e0e0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 800px; margin: 10px auto; box-shadow: 0 4px 14px rgba(0,0,0,0.06);">
                <!-- Header -->
                <div style="border-bottom: 2px solid #1E88E5; padding-bottom: 15px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h2 style="margin: 0; color: #1E88E5; font-size: 1.6rem;">AI ATTENDANCE SYSTEM</h2>
                        <p style="margin: 4px 0 0 0; color: #666; font-size: 0.9rem;">Official Employee Salary Voucher & Payslip</p>
                    </div>
                    <div style="text-align: right;">
                        <span style="background: #E3F2FD; color: #1565C0; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 0.85rem;">CONFIDENTIAL</span>
                        <p style="margin: 6px 0 0 0; color: #888; font-size: 0.85rem;">Date: {issue_date}</p>
                    </div>
                </div>

                <!-- Employee & Shift Details -->
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px; background: #F8F9FA; padding: 15px; border-radius: 8px;">
                    <div>
                        <p style="margin: 3px 0;"><strong>Employee Name:</strong> {chosen_pay['employee_name']}</p>
                        <p style="margin: 3px 0;"><strong>Employee ID:</strong> {chosen_pay['employee_id']}</p>
                        <p style="margin: 3px 0;"><strong>Department:</strong> {chosen_pay['department']}</p>
                        <p style="margin: 3px 0;"><strong>Email:</strong> {chosen_pay['email'] or 'N/A'}</p>
                    </div>
                    <div>
                        <p style="margin: 3px 0;"><strong>Pay Period:</strong> {chosen_pay['period_start']} to {chosen_pay['period_end']}</p>
                        <p style="margin: 3px 0;"><strong>Hourly Wage Rate:</strong> ₹{chosen_pay.get('hourly_rate', 100):.2f} / hr</p>
                        <p style="margin: 3px 0;"><strong>Standard Shift:</strong> 9:00 AM – 5:00 PM (8 hrs = ₹800.00)</p>
                        <p style="margin: 3px 0;"><strong>Payment Status:</strong> <span style="color: #2E7D32; font-weight: 600;">CALCULATED / APPROVED</span></p>
                    </div>
                </div>

                <!-- Attendance & Earnings Breakdown Table -->
                <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 0.95rem;">
                    <thead>
                        <tr style="background: #1E88E5; color: #ffffff; text-align: left;">
                            <th style="padding: 10px 12px;">Compensation Item</th>
                            <th style="padding: 10px 12px; text-align: center;">Hours / Days</th>
                            <th style="padding: 10px 12px; text-align: right;">Hourly Rate</th>
                            <th style="padding: 10px 12px; text-align: right;">Total Amount (₹)</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="border-bottom: 1px solid #eeeeee;">
                            <td style="padding: 10px 12px;"><strong>Total Time Available in Company</strong><br><small style="color: #666;">(Time elapsed between Check-In and Check-Out)</small></td>
                            <td style="padding: 10px 12px; text-align: center;"><strong>{chosen_pay['formatted_hours']}</strong> ({chosen_pay['total_hours_worked']:.2f} hrs)</td>
                            <td style="padding: 10px 12px; text-align: right;">₹{chosen_pay.get('hourly_rate', 100):.2f}/hr</td>
                            <td style="padding: 10px 12px; text-align: right; color: #1565C0; font-weight: 700;">₹{chosen_pay['net_salary']:,.2f}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #eeeeee; background: #FAFAFA;">
                            <td style="padding: 10px 12px;">On-Time Morning Check-Ins (09:00 – 09:30 AM)</td>
                            <td style="padding: 10px 12px; text-align: center;"><strong>{chosen_pay['full_days']}</strong> Days</td>
                            <td style="padding: 10px 12px; text-align: right;">-</td>
                            <td style="padding: 10px 12px; text-align: right; color: #2E7D32;">Logged</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #eeeeee;">
                            <td style="padding: 10px 12px;">Late Morning Check-Ins (After 09:30 AM - Half Day)</td>
                            <td style="padding: 10px 12px; text-align: center;"><strong>{chosen_pay['half_days']}</strong> Days</td>
                            <td style="padding: 10px 12px; text-align: right;">-</td>
                            <td style="padding: 10px 12px; text-align: right; color: #F57C00;">Logged</td>
                        </tr>
                        <tr style="border-bottom: 2px solid #333; background: #FAFAFA;">
                            <td style="padding: 10px 12px;">Absent / Unrecorded Days</td>
                            <td style="padding: 10px 12px; text-align: center; color: #D32F2F;"><strong>{chosen_pay['absent_days']}</strong> Days</td>
                            <td style="padding: 10px 12px; text-align: right;">-</td>
                            <td style="padding: 10px 12px; text-align: right;">₹0.00</td>
                        </tr>
                    </tbody>
                </table>

                <!-- Salary Summary Card -->
                <div style="background: #E8F5E9; border: 1px solid #A5D6A7; padding: 15px 20px; border-radius: 8px; display: flex; justify-content: space-between; align-items: center; margin-bottom: 25px;">
                    <div>
                        <h3 style="margin: 0; color: #2E7D32;">TOTAL MONTHLY NET SALARY</h3>
                        <p style="margin: 4px 0 0 0; color: #555; font-size: 0.85rem;">Total of all daily net salaries (Hours Worked × ₹100 per day)</p>
                    </div>
                    <div style="font-size: 1.8rem; font-weight: 700; color: #1B5E20;">
                        ₹{chosen_pay['net_salary']:,.2f}
                    </div>
                </div>

                <!-- Footer Signatures -->
                <div style="display: flex; justify-content: space-between; margin-top: 40px; padding-top: 15px; border-top: 1px dashed #ccc; font-size: 0.85rem; color: #666;">
                    <div>
                        <p style="margin: 0;">_____________________________</p>
                        <p style="margin: 4px 0 0 0;">Employee Signature</p>
                    </div>
                    <div style="text-align: right;">
                        <p style="margin: 0;">_____________________________</p>
                        <p style="margin: 4px 0 0 0;">Authorized HR / Director</p>
                    </div>
                </div>
            </div>
            """
            clean_slip_html = textwrap.dedent(slip_html).strip()
            components.html(clean_slip_html, height=720, scrolling=True)

            # Download Slip as HTML
            slip_file_slug = f"payslip_{chosen_pay['employee_id']}_{pay_period_choice.lower().replace(' ', '_')}_{selected_year}" if pay_period_choice != "Custom Date Range" else f"payslip_{chosen_pay['employee_id']}_{p_start}_to_{p_end}"
            st.download_button(
                label=f"📥 Download Printable Payslip for {chosen_pay['employee_name']}",
                data=clean_slip_html.encode("utf-8"),
                file_name=f"{slip_file_slug}.html",
                mime="text/html",
                use_container_width=True
            )

            # Expandable Daily Attendance Breakdown
            with st.expander("🔍 View Itemized Daily Attendance & Daily Net Salary Logs"):
                daily_display = []
                for r in chosen_pay["daily_records"]:
                    daily_display.append({
                        "Date": r["date"],
                        "Day": r["day_name"],
                        "Check In": r["check_in"],
                        "Check Out": r["check_out"],
                        "Duration": r["duration"],
                        "Hours": f"{r['hours']:.2f}h" if r['hours'] > 0 else "-",
                        "Status": r["status"],
                        "Daily Net Salary (Hours × ₹100)": f"₹{r['salary_earned']:.2f}",
                        "Notes": r["notes"]
                    })
                daily_df = pd.DataFrame(daily_display)
                st.dataframe(daily_df, use_container_width=True, hide_index=True)

    # Tab 3: Salary Settings & Overrides
    with pay_tabs[2]:
        st.subheader("⚙️ Employee Hourly Wage Settings")
        st.caption("View and configure hourly rates for each registered employee. (Default: ₹100.00 / hr).")
        all_e = get_all_employees()
        if all_e:
            e_rates = pd.DataFrame([{
                "ID": x["employee_id"],
                "Name": x["employee_name"],
                "Department": x["department"],
                "Hourly Rate (₹/hr)": f"₹{float(x.get('hourly_rate') or 100):.2f} / hr",
                "Equivalent 8h Day Rate (₹)": f"₹{float(x.get('daily_salary') or 800):.2f} / day"
            } for x in all_e])
            st.dataframe(e_rates, use_container_width=True, hide_index=True)

            st.markdown("#### Adjust Employee Hourly Rate")
            with st.form("quick_salary_update"):
                target_emp_choice = st.selectbox("Select Employee", [f"{x['employee_id']} - {x['employee_name']}" for x in all_e])
                target_emp_id = target_emp_choice.split(" - ")[0]
                emp_curr = get_employee(target_emp_id)
                new_hourly_rate = st.number_input("New Hourly Wage Rate (₹/hr)", min_value=10.0, max_value=5000.0,
                                                  value=float(emp_curr.get("hourly_rate") or 100.0), step=10.0)
                save_rate = st.form_submit_button("Update Hourly Wage Rate")
                if save_rate:
                    u_ok, u_msg = update_employee(target_emp_id, emp_curr["employee_name"], emp_curr["department"],
                                                  emp_curr.get("email",""), hourly_rate=new_hourly_rate,
                                                  daily_salary=new_hourly_rate * 8.0)
                    if u_ok:
                        st.success(f"Updated {emp_curr['employee_name']}'s wage rate to ₹{new_hourly_rate:.2f}/hr!")
                        st.rerun()
                    else:
                        st.error(u_msg)

# ----------------- PAGE 6: AI MODEL TRAINER -----------------
elif menu == "🧠 AI Model Trainer":
    st.markdown('<div class="main-title">🧠 AI Model Trainer & Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Train OpenCV LBPH recognizer and inspect employee facial training samples</div>', unsafe_allow_html=True)

    t_col1, t_col2 = st.columns([1, 1])

    with t_col1:
        st.markdown("### ⚡ Train Face Recognizer")
        st.write("Trains the Local Binary Patterns Histograms (LBPH) engine on all images located inside `employee_images/`.")

        if st.button("🚀 Train / Retrain Model Now", type="primary"):
            with st.spinner("Extracting facial features and training model..."):
                result = train_model()
            if result["success"]:
                st.success(result["message"])
                st.json(result["labels"])
            else:
                st.error(f"Training Failed: {result['message']}")

    with t_col2:
        st.markdown("### 📊 Model Diagnostics")
        model_exists = TRAINER_MODEL_PATH.exists()
        labels_exist = LABELS_PATH.exists()

        st.write(f"- **Haar Cascade File:** `{'Found' if HAAR_CASCADE_PATH.exists() else 'Missing'}`")
        st.write(f"- **Model Path:** `{TRAINER_MODEL_PATH}`")
        st.write(f"- **Model Trained:** `{'Yes' if model_exists else 'No'}`")
        if model_exists:
            size_mb = os.path.getsize(TRAINER_MODEL_PATH) / (1024 * 1024)
            st.write(f"- **Model File Size:** `{size_mb:.2f} MB`")
            mod_time = datetime.fromtimestamp(os.path.getmtime(TRAINER_MODEL_PATH)).strftime("%Y-%m-%d %H:%M:%S")
            st.write(f"- **Last Trained:** `{mod_time}`")

        if labels_exist:
            with open(LABELS_PATH, "r") as f:
                labels_dict = json.load(f)
            st.write(f"- **Classes Trained:** `{len(labels_dict)}`")

    st.divider()

    st.subheader("🖼️ Face Dataset Explorer")
    st.caption("Inspect collected face training crops for each registered employee.")

    if DATASET_DIR.exists():
        emp_folders = [f.name for f in DATASET_DIR.iterdir() if f.is_dir()]
        if emp_folders:
            selected_folder = st.selectbox("Select Employee Dataset", sorted(emp_folders))
            folder_path = DATASET_DIR / selected_folder
            img_list = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png"))
            st.write(f"Total training images in `{selected_folder}`: **{len(img_list)}**")

            # Preview first 12 images in a 6-column grid
            preview_cols = st.columns(6)
            for idx, img_p in enumerate(img_list[:12]):
                with preview_cols[idx % 6]:
                    st.image(str(img_p), caption=f"#{idx+1}", width=100)
        else:
            st.info("No employee dataset folders found in `employee_images/`.")
    else:
        st.info("Dataset directory not found.")

# ----------------- PAGE 6: SYSTEM SETTINGS -----------------
elif menu == "⚙️ System Settings":
    st.markdown('<div class="main-title">⚙️ System Configuration & Tools</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Fine-tune recognition parameters, work schedules, and manage database backups</div>', unsafe_allow_html=True)

    set_col1, set_col2 = st.columns(2)

    with set_col1:
        st.markdown("### 🛠️ Work Shift & Payroll Policy")
        st.info("⏰ **Daily Hours:** 8 Hours (09:00 AM – 05:00 PM)\n- Check-in: 09:00 – 09:30 AM (Full Day = ₹500)\n- After 09:30 AM: Half Day = ₹250\n- Check-out: 05:00 – 05:30 PM")
        st.slider("Default Confidence Threshold (LBPH Distance)", 20, 65, DEFAULT_CONFIDENCE_THRESHOLD,
                  help="Lower means stricter matching (prevents false matches on unregistered faces); recommended: 40 - 45.")
        st.number_input("Default Base Daily Salary (₹)", value=DEFAULT_DAILY_SALARY, disabled=True,
                        help="Standard daily salary for 8-hour shift.")
        st.number_input("Default Camera Index", min_value=0, max_value=5, value=DEFAULT_CAMERA_INDEX,
                        help="0 is typically the built-in webcam. 1 or 2 for external USB cameras.")

    with set_col2:
        st.markdown("### 🗄️ Database Backup & Maintenance")
        if DB_PATH.exists():
            with open(DB_PATH, "rb") as f:
                db_bytes = f.read()
            st.download_button(
                label="📥 Download Database Backup (.db)",
                data=db_bytes,
                file_name=f"attendance_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db",
                mime="application/x-sqlite3"
            )

        st.markdown("#### 🧹 Database Reset")
        st.warning("Clearing the database will erase all employee and attendance records.")
        confirm_clean = st.checkbox("I understand and want to reset the database tables.")
        if st.button("Reset Database Tables", type="secondary", disabled=not confirm_clean):
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM attendance")
            cursor.execute("DELETE FROM employees")
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='attendance'")
            conn.commit()
            conn.close()
            st.success("Database has been reset successfully.")
            st.rerun()

    st.divider()
    st.markdown("### 📖 About AI Attendance System")
    st.markdown("""
    - **Architecture:** OpenCV LBPH Face Recognizer + Haar Cascade Classifier + SQLite + Streamlit + FastAPI
    - **Developed with:** Modern Python 3, Plotly interactive graphics, and Pandas data analytics
    - **Version:** 2.0.0
    """)
