import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import urllib.request

# Page configuration
st.set_page_config(
    page_title="AI Crowd Monitoring System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Imports
from utils.detection import load_yolo_model, process_frame
from utils.alerts import play_audio_alert, trigger_external_alerts
from utils.logger import log_crowd_data, get_historical_logs, compute_log_metrics, clear_logs

# Inject custom modern glassmorphic styling & keyframe animations
st.markdown("""
    <style>
    /* Premium Title styling */
    .app-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 800;
        background: linear-gradient(45deg, #FF4B4B, #FF8F8F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    .app-subtitle {
        color: #888888;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }
    /* Pulse alert animation */
    @keyframes pulse {
        0% { background-color: rgba(231, 76, 60, 0.15); border-color: rgba(231, 76, 60, 0.4); }
        50% { background-color: rgba(231, 76, 60, 0.55); border-color: rgba(231, 76, 60, 1.0); box-shadow: 0 0 15px rgba(231, 76, 60, 0.4); }
        100% { background-color: rgba(231, 76, 60, 0.15); border-color: rgba(231, 76, 60, 0.4); }
    }
    .overcrowded-alert {
        padding: 18px;
        border-radius: 10px;
        border: 2px solid #e74c3c;
        animation: pulse 1.8s infinite;
        color: #ffffff;
        font-weight: 700;
        text-align: center;
        font-size: 1.3rem;
        margin-bottom: 20px;
        font-family: 'Inter', sans-serif;
    }
    /* Metric Card styling */
    .metric-card {
        background-color: #1E222B;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #FF4B4B;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .metric-title {
        color: #888888;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to download demo video
def download_demo_video():
    demo_url = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/people-detection.mp4"
    demo_path = os.path.join("videos", "demo_crowd.mp4")
    
    if not os.path.exists(demo_path):
        os.makedirs("videos", exist_ok=True)
        with st.spinner("Downloading high-quality demo video (approx. 3MB)..."):
            try:
                urllib.request.urlretrieve(demo_url, demo_path)
            except Exception as e:
                st.error(f"Failed to download demo video: {e}")
                return None
    return demo_path

# Initialize Session States
if "detection_running" not in st.session_state:
    st.session_state.detection_running = False
if "current_people_count" not in st.session_state:
    st.session_state.current_people_count = 0
if "density_status" not in st.session_state:
    st.session_state.density_status = "Low"
if "overcrowded_active" not in st.session_state:
    st.session_state.overcrowded_active = False
if "alert_logs" not in st.session_state:
    st.session_state.alert_logs = []

# Sidebar Navigation / Settings
st.sidebar.markdown("<h2 style='color: #FF4B4B;'>⚙️ SYSTEM CONTROL</h2>", unsafe_allow_html=True)

# Source Selection
source_type = st.sidebar.selectbox(
    "Select Video Input Source",
    ["Demo Video", "Webcam / Live feed", "Upload Video File"]
)

# Detection parameters
st.sidebar.markdown("### 🔍 Model Options")
conf_threshold = st.sidebar.slider("YOLOv8 Confidence Threshold", 0.1, 1.0, 0.35, 0.05)
model_version = st.sidebar.selectbox("YOLOv8 Model Scale", ["yolov8n.pt (Fastest)", "yolov8s.pt (Balanced)"])
selected_model_pt = model_version.split(" ")[0]

# Density thresholds
st.sidebar.markdown("### 🚨 Density Thresholds")
limit_threshold = st.sidebar.number_input("Overcrowding Limit (Trigger Alert)", min_value=1, max_value=200, value=8)
medium_threshold = st.sidebar.number_input("Medium Density Threshold", min_value=1, max_value=100, value=4)

# Overlay settings
st.sidebar.markdown("### 🛠️ Display & Privacy overlays")
enable_blur = st.sidebar.checkbox("Blur Faces (Privacy Mode)", value=False)
enable_grid = st.sidebar.checkbox("Show Zone Grids", value=False)
enable_heatmap = st.sidebar.checkbox("Show Heatmap Overlay", value=False)

# External Alert Settings
st.sidebar.markdown("### 📬 Notification Integrations")
enable_sound = st.sidebar.checkbox("Play Alarm Sounds in Browser", value=True)

telegram_enabled = st.sidebar.checkbox("Telegram Alert Integration", value=False)
telegram_token = ""
telegram_chat_id = ""
if telegram_enabled:
    telegram_token = st.sidebar.text_input("Telegram Bot Token", type="password", help="Enter bot token from BotFather")
    telegram_chat_id = st.sidebar.text_input("Telegram Chat ID", help="Chat ID where alert notifications will be sent")

email_enabled = st.sidebar.checkbox("Gmail Alert Integration", value=False)
email_sender = ""
email_pwd = ""
email_receiver = ""
if email_enabled:
    email_sender = st.sidebar.text_input("Sender Gmail Address")
    email_pwd = st.sidebar.text_input("Gmail App Password", type="password", help="Generate an App Password from Gmail Account Security settings")
    email_receiver = st.sidebar.text_input("Recipient Email Address")

# Load YOLO model
model = load_yolo_model(selected_model_pt)

# Header Section
st.markdown("<h1 class='app-title'>🚨 AI Crowd Monitoring System</h1>", unsafe_allow_html=True)
st.markdown("<p class='app-subtitle'>Real-time vision monitoring, density heatmaps, privacy blurring, and automated alerting protocols.</p>", unsafe_allow_html=True)

# Main Application Tabs
tab_monitor, tab_analytics, tab_config = st.tabs(["📺 Live Monitor", "📊 Analytics Dashboard", "📨 Notifications & System Logs"])

# ==================== TAB 1: LIVE MONITOR ====================
with tab_monitor:
    col_left, col_right = st.columns([7, 3])
    
    with col_left:
        # Stream frame placeholder
        frame_placeholder = st.empty()
        
        # Start/Stop Controls
        col_c1, col_c2, _ = st.columns([2, 2, 6])
        with col_c1:
            if st.button("▶️ Start Monitoring", use_container_width=True):
                st.session_state.detection_running = True
        with col_c2:
            if st.button("⏹️ Stop Monitoring", use_container_width=True):
                st.session_state.detection_running = False
                
        # Handle source loading
        video_path = None
        webcam_active = False
        
        if source_type == "Demo Video":
            video_path = download_demo_video()
        elif source_type == "Upload Video File":
            uploaded_file = st.file_uploader("Upload Video File (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])
            if uploaded_file:
                # Save to a temporary file
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_file.read())
                video_path = tfile.name
        elif source_type == "Webcam / Live feed":
            webcam_active = True
            
    with col_right:
        # Metrics Panel
        st.markdown("<h3 style='margin-top: 0;'>Real-time Status</h3>", unsafe_allow_html=True)
        
        # Bins for KPI Display
        kpi_count = st.empty()
        kpi_density = st.empty()
        kpi_status = st.empty()
        
        # Pulse warning slot
        alert_placeholder = st.empty()
        
        # Integrations status log
        st.markdown("#### System Actions")
        actions_placeholder = st.empty()
        
    # Process Loop
    if st.session_state.detection_running:
        cap = None
        if webcam_active:
            # Safe webcam initialization
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("⚠️ Webcam could not be initialized. Please check connection or verify browser access.")
                st.session_state.detection_running = False
        elif video_path:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("⚠️ Failed to open video file.")
                st.session_state.detection_running = False
        else:
            st.warning("⚠️ Please upload a video file or select Demo Video to start.")
            st.session_state.detection_running = False
            
        if cap and cap.isOpened():
            # Heatmap buffer initialization
            ret, first_frame = cap.read()
            if ret:
                h_shape, w_shape, _ = first_frame.shape
                heatmap_buffer = np.zeros((h_shape, w_shape), dtype=np.float32)
                
                # Rewind cap to start
                if not webcam_active:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                heatmap_buffer = None
                
            # Process frames
            while cap.isOpened() and st.session_state.detection_running:
                ret, frame = cap.read()
                if not ret:
                    if not webcam_active:
                        # Loop video files
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        st.error("Lost video source connection.")
                        break
                        
                # Process the frame using detection utility
                processed_img, count, grid_counts = process_frame(
                    frame=frame,
                    model=model,
                    conf_threshold=conf_threshold,
                    enable_blur=enable_blur,
                    enable_grid=enable_grid,
                    enable_heatmap=enable_heatmap,
                    heatmap_buffer=heatmap_buffer,
                    grid_cols=3,
                    grid_rows=3
                )
                
                # Update Session Metrics
                st.session_state.current_people_count = count
                
                # Calculate overall density rating
                if count == 0:
                    density_level = "Low"
                    density_color = "green"
                elif count <= medium_threshold:
                    density_level = "Low"
                    density_color = "green"
                elif count <= limit_threshold:
                    density_level = "Medium"
                    density_color = "orange"
                else:
                    density_level = "High"
                    density_color = "red"
                    
                st.session_state.density_status = density_level
                overcrowded = count > limit_threshold
                st.session_state.overcrowded_active = overcrowded
                
                # Log stats to CSV (throttled in utils/logger.py)
                log_crowd_data(
                    count=count,
                    threshold=limit_threshold,
                    density_level=density_level,
                    overcrowded=overcrowded,
                    grid_data=grid_counts
                )
                
                # Update Streamlit frame container
                frame_placeholder.image(processed_img, channels="BGR", use_container_width=True)
                
                # Update KPIs in Right Column
                kpi_count.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-title'>👤 Current Count</div>
                        <div class='metric-value'>{count} People</div>
                    </div>
                """, unsafe_allow_html=True)
                
                kpi_density.markdown(f"""
                    <div class='metric-card' style='border-left-color: {density_color};'>
                        <div class='metric-title'>📊 Density Level</div>
                        <div class='metric-value' style='color: {density_color};'>{density_level}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                kpi_status.markdown(f"""
                    <div class='metric-card' style='border-left-color: {"#FF4B4B" if overcrowded else "#2ECC71"};'>
                        <div class='metric-title'>⚠️ Overcrowd Status</div>
                        <div class='metric-value' style='color: {"#FF4B4B" if overcrowded else "#2ECC71"};'>
                            {"ALERT ACTIVE" if overcrowded else "SAFE"}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Handle Alert Overcrowd Event
                if overcrowded:
                    # Pulsing CSS Red Alert Card
                    alert_placeholder.markdown("""
                        <div class="overcrowded-alert">
                            ⚠️ OVERCROWDING WARNING! <br>
                            Count has exceeded the safety limit!
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # 1. Play audio alarm
                    if enable_sound:
                        play_audio_alert()
                        
                    # 2. Trigger External Messaging Logs (Throttled per incident)
                    alert_msg = f"⚠️ SYSTEM ALERT: Overcrowded region detected. Count: {count} (Limit: {limit_threshold}) at {time.strftime('%Y-%m-%d %H:%M:%S')}"
                    settings_dict = {
                        "telegram_enabled": telegram_enabled,
                        "telegram_token": telegram_token,
                        "telegram_chat_id": telegram_chat_id,
                        "email_enabled": email_enabled,
                        "email_sender": email_sender,
                        "email_password": email_pwd,
                        "email_receiver": email_receiver
                    }
                    action_logs = trigger_external_alerts(alert_msg, settings_dict)
                    if action_logs:
                        st.session_state.alert_logs.extend(action_logs)
                        # Render status
                        status_str = ""
                        for app, res in action_logs[-3:]: # Show last 3 alert statuses
                            status_str += f"**{app}**: {res}\n\n"
                        actions_placeholder.markdown(status_str)
                else:
                    alert_placeholder.empty()
                    actions_placeholder.empty()
                    
                # Small Sleep to allow app rendering and stop responsive button clicks
                time.sleep(0.03)
                
            cap.release()
            
    else:
        # Default state when not running
        frame_placeholder.info("📺 Click 'Start Monitoring' above to initialize camera or video feed processing.")
        
        # Draw blank placeholders
        kpi_count.markdown("""
            <div class='metric-card' style='border-left-color: #888888;'>
                <div class='metric-title'>👤 Current Count</div>
                <div class='metric-value'>-- People</div>
            </div>
        """, unsafe_allow_html=True)
        
        kpi_density.markdown("""
            <div class='metric-card' style='border-left-color: #888888;'>
                <div class='metric-title'>📊 Density Level</div>
                <div class='metric-value' style='color: #888888;'>--</div>
            </div>
        """, unsafe_allow_html=True)
        
        kpi_status.markdown("""
            <div class='metric-card' style='border-left-color: #888888;'>
                <div class='metric-title'>⚠️ Overcrowd Status</div>
                <div class='metric-value' style='color: #888888;'>INACTIVE</div>
            </div>
        """, unsafe_allow_html=True)


# ==================== TAB 2: ANALYTICS DASHBOARD ====================
with tab_analytics:
    st.markdown("### 📊 Historical Crowd Data & Charts")
    
    # Load logs
    df_logs = get_historical_logs()
    
    if not df_logs.empty:
        # Get overall metrics
        metrics = compute_log_metrics(df_logs)
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("📈 Peak Crowd Count", f"{metrics['peak_count']} People")
        with col_m2:
            st.metric("⚖️ Average Crowd Count", f"{metrics['average_count']} People")
        with col_m3:
            st.metric("🚨 Total System Alerts", f"{metrics['total_alerts']} Triggers")
        with col_m4:
            st.metric("🔥 High Density Duration", f"{metrics['high_density_pct']}%")
            
        # Draw Plotly Charts
        st.markdown("---")
        
        # 1. Line plot: Crowd size over time
        st.markdown("#### 📈 Crowd Trend Over Time")
        fig_trend = px.line(
            df_logs, 
            x="Timestamp", 
            y=["People Count", "Limit Threshold"],
            color_discrete_map={"People Count": "#FF4B4B", "Limit Threshold": "#FFBF00"},
            title="Registered People Count vs. Safety Threshold Limit"
        )
        fig_trend.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#FAFAFA"
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # 2. Split Charts: Density Distribution & Grid counts
        col_chart_left, col_chart_right = st.columns(2)
        
        with col_chart_left:
            st.markdown("#### 🍩 Density Level Share")
            density_counts = df_logs["Density Level"].value_counts().reset_index()
            density_counts.columns = ["Density Level", "Count"]
            
            fig_pie = px.pie(
                density_counts, 
                values="Count", 
                names="Density Level", 
                color="Density Level",
                color_discrete_map={"Low": "#2ECC71", "Medium": "#F1C40F", "High": "#E74C3C"},
                hole=0.4
            )
            fig_pie.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#FAFAFA"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_chart_right:
            st.markdown("#### ⚠️ Overcrowded Incidents Logged")
            # Overcrowded vs Safe duration
            overcrowd_counts = df_logs["Overcrowded"].value_counts().reset_index()
            overcrowd_counts.columns = ["Status", "Frequency"]
            overcrowd_counts["Status"] = overcrowd_counts["Status"].map({1: "Alarm Triggered", 0: "Safe Zone Status"})
            
            fig_bar = px.bar(
                overcrowd_counts,
                x="Status",
                y="Frequency",
                color="Status",
                color_discrete_map={"Alarm Triggered": "#E74C3C", "Safe Zone Status": "#2ECC71"}
            )
            fig_bar.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#FAFAFA"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
    else:
        st.info("ℹ️ No historical log entries found. Start the live camera feed monitoring loop to generate data metrics.")


# ==================== TAB 3: NOTIFICATIONS & SYSTEM LOGS ====================
with tab_config:
    st.markdown("### 📂 System Activity CSV Logs")
    
    df_logs = get_historical_logs()
    
    if not df_logs.empty:
        # Download log button
        csv_data = df_logs.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Logs to CSV File",
            data=csv_data,
            file_name="crowd_monitor_export.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Display Logs Dataframe
        st.dataframe(df_logs.sort_values(by="Timestamp", ascending=False), use_container_width=True)
        
        # Clear Logs Button
        if st.button("🗑️ Reset and Clear All Saved Logs", type="secondary"):
            if clear_logs():
                st.success("All historical logs deleted successfully!")
                st.rerun()
            else:
                st.error("Failed to delete log files. Verify permissions.")
                
    else:
        st.info("No logs present. Start monitoring first.")
        
    st.markdown("---")
    st.markdown("### 📢 Active Integration Alert Dispatches")
    if st.session_state.alert_logs:
        for app, log_txt in reversed(st.session_state.alert_logs):
            st.markdown(f"**{app}** - {log_txt}")
    else:
        st.write("No external messages sent during this session.")
