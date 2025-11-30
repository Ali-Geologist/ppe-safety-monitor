import streamlit as st
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import time
import os
from pathlib import Path
import tempfile
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import io
import requests
import re
import json

# Set page configuration with professional theme
st.set_page_config(
    page_title="PPE Safety Monitor - Oil & Gas",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .warning-card {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .success-card {
        background-color: #d1edff;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .critical-alert {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 0.5rem;
        animation: blink 1s infinite;
    }
    @keyframes blink {
        50% { opacity: 0.7; }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'violations' not in st.session_state:
    st.session_state.violations = []
if 'monitoring' not in st.session_state:
    st.session_state.monitoring = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'selected_ppe' not in st.session_state:
    st.session_state.selected_ppe = {}
if 'detection_settings' not in st.session_state:
    st.session_state.detection_settings = {
        'confidence': 0.6,
        'speed': 'medium',
        'frame_skip': 2
    }
if 'camera_urls' not in st.session_state:
    st.session_state.camera_urls = []
if 'projects' not in st.session_state:
    st.session_state.projects = {}
if 'current_project' not in st.session_state:
    st.session_state.current_project = None
if 'compliance_threshold' not in st.session_state:
    st.session_state.compliance_threshold = 95.0

# Oil & Gas specific PPE configurations
OIL_GAS_PPE_PRESETS = {
    "Standard Rig Operation": {
        "required": ["Hard Hat", "Safety Glasses", "Safety Boots", "FR Clothing", "Safety Gloves"],
        "optional": ["Hearing Protection", "Face Shield", "Harness"]
    },
    "Drilling Operation": {
        "required": ["Hard Hat", "Face Shield", "FR Clothing", "Safety Boots", "Cut-Resistant Gloves"],
        "optional": ["Hearing Protection", "Harness", "Respirator"]
    },
    "Chemical Handling": {
        "required": ["Chemical Suit", "Respirator", "Chemical Gloves", "Safety Glasses", "Safety Boots"],
        "optional": ["Face Shield", "Apron"]
    },
    "Confined Space": {
        "required": ["Harness", "Hard Hat", "Safety Boots", "Headlamp", "Gas Monitor"],
        "optional": ["Respirator", "Communication Device"]
    },
    "Hot Work": {
        "required": ["FR Clothing", "Face Shield", "Welding Gloves", "Safety Boots", "Hard Hat"],
        "optional": ["Apron", "Respirator"]
    },
    "Custom Project": {
        "required": [],
        "optional": []
    }
}

# Load model with enhanced caching and error handling
@st.cache_resource(show_spinner="Loading AI Safety Model...")
def load_model():
    try:
        model = YOLO(r"D:\runs\detect\train\weights\best.pt")
        st.success("✅ Safety Model Loaded Successfully!")
        
        # Verify model classes
        if hasattr(model, 'names'):
            st.info(f"📋 Model can detect {len(model.names)} safety items")
            
        return model
    except Exception as e:
        st.error(f"❌ Model Loading Failed: {e}")
        st.info("💡 Please ensure your model file is accessible and in the correct format")
        return None

def get_available_classes(model):
    """Get available class names from the model"""
    if model and hasattr(model, 'names'):
        return model.names
    return {}

def validate_ip_camera_url(url):
    """Enhanced IP camera URL validation"""
    if not url:
        return False, "URL cannot be empty"
    
    # Enhanced URL validation for oil & gas environments
    ip_pattern = r'^rtsp://|^http://|^https://'
    if not re.match(ip_pattern, url):
        return False, "URL must start with rtsp://, http://, or https://"
    
    # Security check for common vulnerabilities
    if '///' in url or '%%' in url:
        return False, "Invalid URL format detected"
    
    return True, "URL validation passed"

def test_ip_camera(url, timeout=8):
    """Enhanced IP camera testing with better diagnostics"""
    try:
        if url.startswith('rtsp://'):
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    return True, f"✅ RTSP camera connected - Resolution: {frame.shape[1]}x{frame.shape[0]}"
                else:
                    return False, "❌ RTSP camera connected but no frame received"
            else:
                return False, "❌ Cannot connect to RTSP stream - Check credentials/network"
        
        elif url.startswith(('http://', 'https://')):
            try:
                response = requests.get(url, timeout=timeout, stream=True)
                if response.status_code == 200:
                    return True, "✅ HTTP camera connected successfully"
                else:
                    return False, f"❌ HTTP camera returned status: {response.status_code}"
            except requests.exceptions.RequestException as e:
                return False, f"❌ HTTP camera connection failed: {e}"
        
        else:
            return False, "❌ Unsupported URL protocol"
    
    except Exception as e:
        return False, f"❌ Camera test failed: {e}"

def main():
    # Professional header with company branding
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown('<h1 class="main-header">🛡️ OIL & GAS PPE SAFETY MONITORING SYSTEM</h1>', unsafe_allow_html=True)
    
    # Status bar
    show_status_bar()
    
    # Sidebar with enhanced navigation
    st.sidebar.title("🔧 Navigation Panel")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Select Module",
        ["Project Setup", "Camera Configuration", "Detection Settings", "Live Monitoring", 
         "Safety Dashboard", "Compliance Reports", "Deployment Guide"]
    )
    
    # Display selected page
    if page == "Project Setup":
        show_project_setup()
    elif page == "Camera Configuration":
        show_camera_setup()
    elif page == "Detection Settings":
        show_detection_settings()
    elif page == "Live Monitoring":
        show_live_monitoring()
    elif page == "Safety Dashboard":
        show_safety_dashboard()
    elif page == "Compliance Reports":
        show_compliance_reports()
    elif page == "Deployment Guide":
        show_deployment_guide()

def show_status_bar():
    """Display professional status bar"""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        status = "🟢 ACTIVE" if st.session_state.monitoring else "🔴 INACTIVE"
        st.metric("Monitoring Status", status)
    
    with col2:
        st.metric("Violations Today", len(st.session_state.violations))
    
    with col3:
        project = st.session_state.current_project or "No Project"
        st.metric("Current Project", project[:15] + "..." if len(project) > 15 else project)
    
    with col4:
        ppe_count = len(st.session_state.selected_ppe)
        st.metric("PPE Items Monitored", ppe_count)
    
    with col5:
        compliance = calculate_compliance_rate()
        st.metric("Compliance Rate", f"{compliance:.1f}%")

def calculate_compliance_rate():
    """Calculate safety compliance rate"""
    if not st.session_state.violations:
        return 100.0
    
    total_checks = len(st.session_state.violations) * 10  # Assuming 10 checks per violation period
    violations_count = len(st.session_state.violations)
    
    if total_checks == 0:
        return 100.0
    
    compliance_rate = ((total_checks - violations_count) / total_checks) * 100
    return min(compliance_rate, 100.0)

def show_project_setup():
    st.markdown('<h2 class="section-header">🏗️ Project Safety Configuration</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Project Information")
        
        project_name = st.text_input("Project Name", placeholder="e.g., North Sea Platform Maintenance")
        project_location = st.text_input("Project Location", placeholder="e.g., Platform A, Deck 3")
        project_supervisor = st.text_input("Safety Supervisor", placeholder="Name of responsible person")
        
        st.subheader("PPE Requirement Presets")
        
        preset_selection = st.selectbox(
            "Select Safety Preset",
            list(OIL_GAS_PPE_PRESETS.keys()),
            help="Choose predefined safety configurations for different operations"
        )
        
        # Display preset details
        preset = OIL_GAS_PPE_PRESETS[preset_selection]
        st.info(f"**{preset_selection}:** {', '.join(preset['required'])}")
        
        if preset['optional']:
            st.info(f"**Recommended:** {', '.join(preset['optional'])}")
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("Safety Briefing")
        st.write("✅ Define project-specific PPE requirements")
        st.write("✅ Configure monitoring parameters")
        st.write("✅ Set compliance thresholds")
        st.write("✅ Assign safety responsibilities")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Compliance threshold
        st.subheader("Compliance Settings")
        compliance_threshold = st.slider(
            "Minimum Compliance Rate (%)",
            min_value=80.0,
            max_value=99.9,
            value=95.0,
            step=0.1,
            help="Set the minimum acceptable safety compliance rate"
        )
        st.session_state.compliance_threshold = compliance_threshold
    
    # Custom PPE Selection
    st.markdown("---")
    st.subheader("Custom PPE Requirements")
    
    if st.session_state.model is None:
        st.session_state.model = load_model()
    
    if st.session_state.model:
        available_classes = get_available_classes(st.session_state.model)
        
        if available_classes:
            col1, col2, col3 = st.columns(3)
            selected_ppe = {}
            
            # Group PPE items by category
            head_protection = {id: name for id, name in available_classes.items() 
                             if any(keyword in name.lower() for keyword in ['helmet', 'hat', 'hardhat'])}
            eye_protection = {id: name for id, name in available_classes.items() 
                            if any(keyword in name.lower() for keyword in ['glass', 'goggle', 'eye'])}
            body_protection = {id: name for id, name in available_classes.items() 
                             if any(keyword in name.lower() for keyword in ['vest', 'clothing', 'fr', 'suit'])}
            hand_protection = {id: name for id, name in available_classes.items() 
                             if any(keyword in name.lower() for keyword in ['glove', 'hand'])}
            foot_protection = {id: name for id, name in available_classes.items() 
                             if any(keyword in name.lower() for keyword in ['boot', 'shoe', 'foot'])}
            respiratory = {id: name for id, name in available_classes.items() 
                         if any(keyword in name.lower() for keyword in ['mask', 'respirator'])}
            fall_protection = {id: name for id, name in available_classes.items() 
                             if any(keyword in name.lower() for keyword in ['harness', 'fall'])}
            
            with col1:
                st.write("**Head Protection**")
                for class_id, class_name in head_protection.items():
                    if st.checkbox(f"{class_name} (Class {class_id})", value=True):
                        selected_ppe[class_id] = class_name
                
                st.write("**Eye Protection**")
                for class_id, class_name in eye_protection.items():
                    if st.checkbox(f"{class_name} (Class {class_id})", value=True):
                        selected_ppe[class_id] = class_name
            
            with col2:
                st.write("**Body Protection**")
                for class_id, class_name in body_protection.items():
                    if st.checkbox(f"{class_name} (Class {class_id})", value=True):
                        selected_ppe[class_id] = class_name
                
                st.write("**Hand Protection**")
                for class_id, class_name in hand_protection.items():
                    if st.checkbox(f"{class_name} (Class {class_id})", value=True):
                        selected_ppe[class_id] = class_name
            
            with col3:
                st.write("**Foot Protection**")
                for class_id, class_name in foot_protection.items():
                    if st.checkbox(f"{class_name} (Class {class_id})", value=True):
                        selected_ppe[class_id] = class_name
                
                st.write("**Respiratory & Fall Protection**")
                for class_id, class_name in {**respiratory, **fall_protection}.items():
                    if st.checkbox(f"{class_name} (Class {class_id})", value=True):
                        selected_ppe[class_id] = class_name
            
            # Save project configuration
            if st.button("💾 Save Project Configuration", type="primary"):
                if project_name and selected_ppe:
                    st.session_state.projects[project_name] = {
                        'name': project_name,
                        'location': project_location,
                        'supervisor': project_supervisor,
                        'ppe_requirements': selected_ppe,
                        'preset': preset_selection,
                        'created_at': datetime.now()
                    }
                    st.session_state.current_project = project_name
                    st.session_state.selected_ppe = selected_ppe
                    st.success(f"✅ Project '{project_name}' configured successfully!")
                    st.info(f"📋 Monitoring {len(selected_ppe)} PPE items for this project")
                else:
                    st.error("❌ Please provide project name and select at least one PPE item")
        else:
            st.error("No classes found in the model. Please check your model configuration.")
    else:
        st.error("Model not loaded. Please check the model path.")

def show_camera_setup():
    st.markdown('<h2 class="section-header">📷 Camera System Configuration</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Local Cameras", "IP Cameras", "Mobile Integration", "Video Files"])
    
    with tab1:
        st.subheader("Local Camera Setup")
        st.info("Configure built-in or USB cameras for local monitoring")
        
        if st.button("🔍 Scan for Local Cameras", type="primary"):
            detect_local_cameras()
    
    with tab2:
        st.subheader("IP Camera Configuration")
        st.info("Connect to network cameras for remote site monitoring")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            camera_url = st.text_input(
                "Camera Stream URL:",
                placeholder="rtsp://username:password@ip:port/stream"
            )
        
        with col2:
            st.write("")
            if st.button("🔗 Test Connection", use_container_width=True):
                if camera_url:
                    with st.spinner("Testing camera connection..."):
                        success, result = test_ip_camera(camera_url)
                    if success:
                        st.success(result)
                        if camera_url not in st.session_state.camera_urls:
                            st.session_state.camera_urls.append(camera_url)
                    else:
                        st.error(result)
                else:
                    st.error("Please enter a camera URL")
        
        # Professional camera templates
        with st.expander("🎯 Oil & Gas Camera Templates"):
            st.markdown("""
            **Offshore Platform Cameras:**
            - Hikvision: `rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101`
            - Axis: `rtsp://root:pass@platform-cam/axis-media/media.amp`
            
            **Drilling Rig Cameras:**
            - Dahua: `rtsp://admin:password@rig-cam:554/cam/realmonitor?channel=1&subtype=0`
            - Bosch: `rtsp://admin@drill-rig:554/stream1`
            
            **Refinery Area Cameras:**
            - RTSP over secure VPN tunnels
            - HTTP streams with authentication
            """)
    
    with tab3:
        st.subheader("Mobile Camera Integration")
        st.info("Use mobile devices as temporary monitoring cameras")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Android Devices:**
            1. Install **IP Webcam** app
            2. Configure security settings
            3. Start server with authentication
            4. Use URL: `http://device-ip:8080/video`
            
            **Security Note:** Use secure networks only
            """)
        
        with col2:
            st.markdown("""
            **iOS Devices:**
            1. Install **IP Camera** app
            2. Enable RTSP streaming
            3. Set up authentication
            4. Use provided secure URL
            
            **Best Practice:** Temporary use only for inspections
            """)
    
    with tab4:
        st.subheader("Video File Analysis")
        st.info("Analyze pre-recorded safety footage")
        
        uploaded_file = st.file_uploader(
            "Upload Safety Video", 
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload recorded footage for compliance analysis"
        )
        
        if uploaded_file:
            st.success(f"✅ Video ready for analysis: {uploaded_file.name}")
            if st.button("📊 Analyze Video Compliance"):
                analyze_video_compliance(uploaded_file)

def show_detection_settings():
    st.markdown('<h2 class="section-header">⚙️ Advanced Detection Settings</h2>', unsafe_allow_html=True)
    
    if not st.session_state.selected_ppe:
        st.warning("⚠️ Please configure project PPE requirements first!")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Performance Configuration")
        
        # Real-time performance settings
        confidence = st.slider(
            "Detection Confidence Threshold",
            min_value=0.3,
            max_value=0.9,
            value=0.6,
            step=0.05,
            help="Higher values reduce false positives but may miss some detections"
        )
        
        processing_mode = st.selectbox(
            "Processing Mode",
            ["High Speed", "Balanced", "High Accuracy"],
            index=1,
            help="Optimize for your hardware and requirements"
        )
        
        frame_skip = st.slider(
            "Frame Processing Rate",
            min_value=1,
            max_value=10,
            value=2,
            help="Process every Nth frame for performance optimization"
        )
        
        # Alert settings
        st.subheader("Alert Configuration")
        
        alert_level = st.selectbox(
            "Violation Alert Level",
            ["Low - Log Only", "Medium - Visual Alert", "High - Audio & Visual"],
            index=1,
            help="Set how violations are reported"
        )
        
        auto_report = st.checkbox(
            "Generate Automatic Compliance Reports",
            value=True,
            help="Automatically generate hourly/daily compliance reports"
        )
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("Current Configuration")
        st.write(f"**Project:** {st.session_state.current_project or 'Not Set'}")
        st.write(f"**PPE Items:** {len(st.session_state.selected_ppe)}")
        st.write(f"**Confidence:** {confidence}")
        st.write(f"**Mode:** {processing_mode}")
        st.write(f"**Frame Skip:** {frame_skip}")
        st.write(f"**Alerts:** {alert_level}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Save settings
        if st.button("💾 Apply Settings", type="primary", use_container_width=True):
            # Map processing modes to parameters
            speed_params = {
                "High Speed": {"imgsz": 320, "half": True},
                "Balanced": {"imgsz": 640, "half": False},
                "High Accuracy": {"imgsz": 1280, "half": False}
            }
            
            st.session_state.detection_settings = {
                'confidence': confidence,
                'speed': processing_mode,
                'frame_skip': frame_skip,
                'speed_params': speed_params[processing_mode],
                'alert_level': alert_level,
                'auto_report': auto_report
            }
            st.success("✅ Settings applied successfully!")

def show_live_monitoring():
    st.markdown('<h2 class="section-header">📹 Live Safety Monitoring</h2>', unsafe_allow_html=True)
    
    if not st.session_state.selected_ppe:
        st.warning("⚠️ Please configure project and PPE requirements first!")
        return
    
    if st.session_state.model is None:
        st.session_state.model = load_model()
    
    # Display current project info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.write("**Current Project**")
        st.write(f"### {st.session_state.current_project or 'Not Set'}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.write("**PPE Requirements**")
        st.write(f"### {len(st.session_state.selected_ppe)} Items")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        compliance = calculate_compliance_rate()
        status_class = "success-card" if compliance >= st.session_state.compliance_threshold else "warning-card"
        st.markdown(f'<div class="{status_class}">', unsafe_allow_html=True)
        st.write("**Compliance Rate**")
        st.write(f"### {compliance:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Monitoring controls
    st.subheader("Monitoring Controls")
    
    source_type = st.radio(
        "Video Source",
        ["Local Camera", "IP Camera", "Test Mode", "Video File"],
        horizontal=True
    )
    
    if source_type == "Local Camera":
        if st.button("🎥 Start Local Monitoring", type="primary", use_container_width=True):
            st.session_state.monitoring = True
            start_webcam_monitoring()
    
    elif source_type == "IP Camera":
        if st.session_state.camera_urls:
            selected_url = st.selectbox("Select Camera", st.session_state.camera_urls)
            if st.button("🌐 Start IP Monitoring", type="primary", use_container_width=True):
                st.session_state.monitoring = True
                start_ip_camera_monitoring(selected_url)
        else:
            st.warning("No IP cameras configured. Please add cameras first.")
    
    elif source_type == "Test Mode":
        if st.button("🧪 Start Test Mode", type="primary", use_container_width=True):
            st.session_state.monitoring = True
            start_test_mode()
    
    # Stop button
    if st.session_state.monitoring:
        if st.button("⏹️ Stop Monitoring", type="secondary", use_container_width=True):
            st.session_state.monitoring = False
            st.rerun()

def show_safety_dashboard():
    st.markdown('<h2 class="section-header">📊 Safety Compliance Dashboard</h2>', unsafe_allow_html=True)
    
    if not st.session_state.violations:
        st.info("📈 No safety data available yet. Start monitoring to see analytics.")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_violations = len(st.session_state.violations)
        st.metric("Total Violations", total_violations)
    
    with col2:
        today_violations = len([v for v in st.session_state.violations 
                              if v['timestamp'].date() == datetime.now().date()])
        st.metric("Today's Violations", today_violations)
    
    with col3:
        compliance_rate = calculate_compliance_rate()
        st.metric("Compliance Rate", f"{compliance_rate:.1f}%")
    
    with col4:
        status = "✅ COMPLIANT" if compliance_rate >= st.session_state.compliance_threshold else "⚠️ NEEDS ATTENTION"
        st.metric("Safety Status", status)
    
    # Charts and analytics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Violations Trend")
        df = pd.DataFrame([
            {
                'timestamp': v['timestamp'],
                'hour': v['timestamp'].hour,
                'missing_ppe': v['missing_ppe']
            }
            for v in st.session_state.violations
        ])
        
        if not df.empty:
            hourly_data = df.groupby('hour').size()
            fig = px.line(
                x=hourly_data.index,
                y=hourly_data.values,
                labels={'x': 'Hour of Day', 'y': 'Violations'},
                title="Violations by Hour"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("PPE Compliance Breakdown")
        ppe_violations = {}
        for violation in st.session_state.violations:
            items = violation['missing_ppe'].split(', ')
            for item in items:
                ppe_violations[item] = ppe_violations.get(item, 0) + 1
        
        if ppe_violations:
            fig = px.pie(
                values=list(ppe_violations.values()),
                names=list(ppe_violations.keys()),
                title="Most Frequently Missing PPE"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Recent violations
    st.subheader("Recent Safety Violations")
    for i, violation in enumerate(st.session_state.violations[-10:]):
        with st.expander(f"🚨 Violation {i+1} - {violation['timestamp'].strftime('%H:%M:%S')}"):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(violation['image'], use_column_width=True)
            with col2:
                st.write(f"**Missing PPE:** {violation['missing_ppe']}")
                st.write(f"**Time:** {violation['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"**Project:** {st.session_state.current_project}")

def show_compliance_reports():
    st.markdown('<h2 class="section-header">📈 Compliance Reporting</h2>', unsafe_allow_html=True)
    
    if not st.session_state.violations:
        st.warning("No compliance data available for reporting.")
        return
    
    # Report generation options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Generate Excel Report", type="primary", use_container_width=True):
            generate_excel_report()
    
    with col2:
        if st.button("📋 Generate PDF Summary", use_container_width=True):
            generate_pdf_summary()
    
    with col3:
        if st.button("🔄 Refresh Analytics", use_container_width=True):
            st.rerun()
    
    # Data table
    st.subheader("Violation Records")
    df = pd.DataFrame([
        {
            'Timestamp': v['timestamp'],
            'Missing PPE': v['missing_ppe'],
            'Project': st.session_state.current_project,
            'Compliance Rate': f"{calculate_compliance_rate():.1f}%"
        }
        for v in st.session_state.violations
    ])
    
    st.dataframe(df, use_container_width=True)
    
    # Export options
    st.download_button(
        "📥 Export CSV",
        df.to_csv(index=False),
        "safety_compliance_report.csv",
        "text/csv"
    )

def show_deployment_guide():
    st.markdown('<h2 class="section-header">🌐 Enterprise Deployment</h2>', unsafe_allow_html=True)
    
    st.info("""
    **Oil & Gas Deployment Considerations:**
    - Offshore platform network constraints
    - Hazardous area compliance
    - Redundant systems for critical monitoring
    - Integration with existing safety systems
    """)
    
    # Deployment options
    tab1, tab2, tab3 = st.tabs(["Cloud Deployment", "On-Premises", "Hybrid"])
    
    with tab1:
        st.subheader("Cloud Deployment Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **AWS for Oil & Gas**
            - Amazon SageMaker for model hosting
            - AWS IoT for camera streams
            - S3 for compliance data storage
            - CloudWatch for monitoring
            
            **Azure Energy Services**
            - Azure IoT Hub for device management
            - Azure ML for AI capabilities
            - Power BI for compliance dashboards
            """)
        
        with col2:
            st.markdown("""
            **Specialized Solutions**
            - **Schlumberger** DELFI integration
            - **Halliburton** DecisionSpace compatibility
            - **Baker Hughes** IIoT platforms
            - **OSIsoft** PI System data integration
            """)
    
    with tab2:
        st.subheader("On-Premises Deployment")
        
        st.markdown("""
        **Offshore Platform Requirements:**
        - Ruggedized servers and networking
        - Redundant power supplies
        - Satellite communication backup
        - Local processing for network outages
        
        **Safety Certifications Required:**
        - ATEX/IECEx for hazardous areas
        - Marine classification society approvals
        - Cybersecurity certifications
        """)
    
    with tab3:
        st.subheader("Hybrid Deployment")
        
        st.markdown("""
        **Best Practice Architecture:**
        - Local processing on platforms/rigs
        - Cloud synchronization when connected
        - Edge AI for real-time detection
        - Centralized compliance reporting
        
        **Data Flow:**
        Platform Cameras → Local Server → Cloud Dashboard → Management Reports
        """)

# Enhanced monitoring functions (similar to original but with professional features)
def start_webcam_monitoring():
    """Enhanced webcam monitoring with professional features"""
    st.info("🚀 Starting professional safety monitoring...")
    
    confidence = st.session_state.detection_settings['confidence']
    frame_skip = st.session_state.detection_settings['frame_skip']
    speed_params = st.session_state.detection_settings['speed_params']
    selected_classes = list(st.session_state.selected_ppe.keys())
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        st.error("❌ Cannot access safety camera. Please check connections.")
        return
    
    # Professional camera configuration
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    run_enhanced_monitoring_loop(cap, "Safety Monitoring Camera", selected_classes, confidence, frame_skip, speed_params)

def start_ip_camera_monitoring(camera_url):
    """Enhanced IP camera monitoring"""
    st.info(f"🌐 Connecting to safety camera: {camera_url}")
    
    confidence = st.session_state.detection_settings['confidence']
    frame_skip = st.session_state.detection_settings['frame_skip']
    speed_params = st.session_state.detection_settings['speed_params']
    selected_classes = list(st.session_state.selected_ppe.keys())
    
    success, message = test_ip_camera(camera_url)
    if not success:
        st.error(f"❌ {message}")
        st.session_state.monitoring = False
        return
    
    st.success("✅ Safety camera connected successfully!")
    
    cap = cv2.VideoCapture(camera_url)
    if not cap.isOpened():
        st.error("❌ Failed to initialize safety camera stream")
        st.session_state.monitoring = False
        return
    
    run_enhanced_monitoring_loop(cap, f"Remote Safety Camera", selected_classes, confidence, frame_skip, speed_params)

def run_enhanced_monitoring_loop(cap, source_name, selected_classes, confidence, frame_skip, speed_params):
    """Professional monitoring loop with enhanced features"""
    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    performance_placeholder = st.empty()
    compliance_placeholder = st.empty()
    
    frame_count = 0
    processing_times = []
    violation_count = 0
    
    while st.session_state.monitoring and cap.isOpened():
        try:
            start_time = time.time()
            ret, frame = cap.read()
            
            if not ret:
                st.error("❌ Lost connection to safety camera")
                break
            
            frame_count += 1
            
            if frame_count % frame_skip == 0:
                # Professional detection with enhanced visualization
                results = st.session_state.model(
                    frame, 
                    conf=confidence,
                    classes=selected_classes,
                    verbose=False,
                    **speed_params
                )
                
                violations = check_for_violations(results, selected_classes)
                annotated_frame = results[0].plot()
                
                # Enhanced professional overlay
                compliance_rate = calculate_compliance_rate()
                status_color = (0, 255, 0) if not violations else (0, 0, 255)
                
                cv2.putText(annotated_frame, f"OIL&GAS SAFETY MONITOR", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Project: {st.session_state.current_project}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(annotated_frame, f"Compliance: {compliance_rate:.1f}%", (10, 85), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
                
                if violations:
                    cv2.putText(annotated_frame, f"VIOLATION: {', '.join(violations)}", (10, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    status_placeholder.error(f"🚨 SAFETY VIOLATION: Missing {', '.join(violations)}")
                    save_violation(frame, violations)
                    violation_count += 1
                else:
                    status_placeholder.success("✅ All safety requirements met")
                
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(annotated_frame_rgb, use_column_width=True)
            
            # Performance monitoring
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            if frame_count % 30 == 0:
                avg_time = np.mean(processing_times[-30:]) if processing_times else 0
                fps = 1.0 / avg_time if avg_time > 0 else 0
                
                performance_placeholder.info(
                    f"**System Performance:** {fps:.1f} FPS | "
                    f"Processing: {avg_time*1000:.1f}ms | "
                    f"Violations: {violation_count}"
                )
                
                compliance_placeholder.metric(
                    "Current Compliance Rate", 
                    f"{compliance_rate:.1f}%",
                    delta=f"Target: {st.session_state.compliance_threshold}%"
                )
            
            time.sleep(0.01)
            
        except Exception as e:
            st.error(f"Monitoring system error: {e}")
            break
    
    cap.release()

def start_test_mode():
    """Enhanced test mode for oil & gas scenarios"""
    st.success("🎯 Professional Test Mode Active - Oil & Gas Safety Simulation")
    
    selected_classes = list(st.session_state.selected_ppe.keys())
    ppe_items = list(st.session_state.selected_ppe.values())
    
    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    scenario_placeholder = st.empty()
    
    frame_count = 0
    scenarios = [
        "Full Compliance - All PPE Present",
        "Missing Head Protection",
        "Missing Eye Protection", 
        "Missing Hand Protection",
        "Multiple Violations",
        "Critical Safety Breach"
    ]
    
    while st.session_state.monitoring:
        try:
            scenario_index = (frame_count // 50) % len(scenarios)
            current_scenario = scenarios[scenario_index]
            scenario_placeholder.info(f"**Training Scenario:** {current_scenario}")
            
            # Create professional test image
            test_image = create_professional_test_image(frame_count, ppe_items, current_scenario)
            
            # Run detection
            results = st.session_state.model(
                test_image, 
                conf=st.session_state.detection_settings['confidence'],
                classes=selected_classes,
                verbose=False
            )
            
            violations = check_for_violations(results, selected_classes)
            annotated_frame = results[0].plot() if results and len(results) > 0 else test_image
            
            # Add professional overlay
            cv2.putText(annotated_frame, "OIL & GAS SAFETY TRAINING", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(annotated_frame, f"Scenario: {current_scenario}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(annotated_frame_rgb, use_column_width=True)
            
            if violations:
                status_placeholder.warning(f"🚨 Training Violation: {', '.join(violations)}")
                if frame_count % 30 == 0:
                    save_violation(test_image, violations)
            else:
                status_placeholder.success("✅ Perfect compliance - Safety standards met")
            
            frame_count += 1
            time.sleep(0.2)
            
        except Exception as e:
            st.error(f"Training system error: {e}")
            break

def create_professional_test_image(frame_count, ppe_items, scenario):
    """Create professional test images for oil & gas training"""
    img = np.ones((720, 1280, 3), dtype=np.uint8) * 50  # Dark background
    
    # Simulate industrial background
    cv2.rectangle(img, (100, 100), (1180, 600), (80, 80, 80), -1)
    cv2.rectangle(img, (100, 100), (1180, 600), (200, 200, 200), 2)
    
    # Person silhouette
    cv2.ellipse(img, (640, 400), (120, 200), 0, 0, 360, (100, 100, 100), -1)
    
    # Scenario-based PPE visualization
    missing_items = []
    if "Missing Head" in scenario:
        missing_items.append("Hard Hat")
    if "Missing Eye" in scenario:
        missing_items.append("Safety Glasses") 
    if "Missing Hand" in scenario:
        missing_items.append("Safety Gloves")
    if "Multiple" in scenario:
        missing_items = ["Hard Hat", "Safety Vest"]
    if "Critical" in scenario:
        missing_items = ppe_items[:3]  # Missing first 3 items
    
    # Draw available PPE
    y_pos = 120
    for i, ppe_item in enumerate(ppe_items):
        if ppe_item not in missing_items:
            color = [(0, 255, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255)][i % 5]
            cv2.rectangle(img, (150, y_pos), (400, y_pos + 30), color, -1)
            cv2.putText(img, f"✓ {ppe_item}", (160, y_pos + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            y_pos += 40
    
    # Draw missing PPE
    if missing_items:
        y_pos = 120
        for i, missing_item in enumerate(missing_items):
            cv2.rectangle(img, (800, y_pos), (1050, y_pos + 30), (0, 0, 255), -1)
            cv2.putText(img, f"✗ {missing_item}", (810, y_pos + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 40
        
        cv2.putText(img, "SAFETY VIOLATION DETECTED", (400, 650), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Add professional header
    cv2.putText(img, "OIL & GAS PPE COMPLIANCE MONITORING", (200, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, f"Project: {st.session_state.current_project}", (200, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img

def analyze_video_compliance(uploaded_file):
    """Enhanced video analysis with professional reporting"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("❌ Cannot open video file for analysis")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    st.info(f"📊 Analyzing {total_frames} frames at {fps:.1f} FPS")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    frame_placeholder = st.empty()
    
    compliance_data = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection every 10th frame for performance
        if frame_count % 10 == 0:
            results = st.session_state.model(
                frame,
                conf=st.session_state.detection_settings['confidence'],
                classes=list(st.session_state.selected_ppe.keys()),
                verbose=False
            )
            
            violations = check_for_violations(results, list(st.session_state.selected_ppe.keys()))
            
            compliance_data.append({
                'frame': frame_count,
                'timestamp': frame_count / fps,
                'violations': len(violations),
                'missing_ppe': ', '.join(violations) if violations else 'None'
            })
            
            # Display sample frames
            if frame_count % 100 == 0:
                annotated_frame = results[0].plot() if results and len(results) > 0 else frame
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(annotated_frame_rgb, caption=f"Frame {frame_count}")
        
        frame_count += 1
        progress_bar.progress(frame_count / total_frames)
        status_text.text(f"Processed {frame_count}/{total_frames} frames")
    
    cap.release()
    os.unlink(video_path)
    
    # Generate compliance report
    if compliance_data:
        df = pd.DataFrame(compliance_data)
        total_violations = df['violations'].sum()
        compliance_rate = ((len(df) - len(df[df['violations'] > 0])) / len(df)) * 100
        
        st.success(f"✅ Video analysis complete!")
        st.metric("Overall Compliance Rate", f"{compliance_rate:.1f}%")
        st.metric("Total Violation Frames", total_violations)
        
        # Show compliance trend
        fig = px.line(df, x='timestamp', y='violations', title="Compliance Trend During Video")
        st.plotly_chart(fig, use_container_width=True)

def check_for_violations(results, required_classes):
    """Enhanced violation detection"""
    detected_classes = set()
    
    if results and len(results) > 0:
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            detected_classes.add(class_id)
    
    missing_ppe = []
    for class_id in required_classes:
        if class_id not in detected_classes:
            ppe_name = st.session_state.selected_ppe.get(class_id, f"Class {class_id}")
            missing_ppe.append(ppe_name)
    
    return missing_ppe

def save_violation(frame, violations):
    """Enhanced violation recording"""
    violation_record = {
        'timestamp': datetime.now(),
        'missing_ppe': ', '.join(violations),
        'image': frame.copy(),
        'project': st.session_state.current_project,
        'compliance_rate': calculate_compliance_rate()
    }
    st.session_state.violations.append(violation_record)

def generate_excel_report():
    """Generate professional Excel compliance report"""
    df = pd.DataFrame([
        {
            'Timestamp': v['timestamp'],
            'Missing_PPE': v['missing_ppe'],
            'Project': v['project'],
            'Compliance_Rate': v['compliance_rate'],
            'Date': v['timestamp'].date(),
            'Time': v['timestamp'].time(),
            'Hour': v['timestamp'].hour
        }
        for v in st.session_state.violations
    ])
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Violations sheet
        df.to_excel(writer, sheet_name='Safety_Violations', index=False)
        
        # Summary sheet
        summary_data = {
            'Report_Generated': [datetime.now()],
            'Total_Violations': [len(df)],
            'Current_Project': [st.session_state.current_project],
            'Overall_Compliance_Rate': [calculate_compliance_rate()],
            'Target_Compliance_Rate': [st.session_state.compliance_threshold],
            'PPE_Items_Monitored': [len(st.session_state.selected_ppe)],
            'Monitoring_Period': [f"{df['Timestamp'].min()} to {df['Timestamp'].max()}" if not df.empty else 'N/A']
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Executive_Summary', index=False)
        
        # Configuration sheet
        config_data = {
            'PPE_Item': list(st.session_state.selected_ppe.values()),
            'Class_ID': list(st.session_state.selected_ppe.keys())
        }
        pd.DataFrame(config_data).to_excel(writer, sheet_name='Safety_Configuration', index=False)
    
    st.download_button(
        "📥 Download Professional Report",
        output.getvalue(),
        f"oil_gas_safety_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

def generate_pdf_summary():
    """Placeholder for PDF report generation"""
    st.info("📋 PDF report generation feature coming soon...")
    # Implementation would use libraries like reportlab or weasyprint

def detect_local_cameras():
    """Enhanced local camera detection"""
    st.info("🔍 Scanning for safety monitoring cameras...")
    
    available_cameras = []
    max_cameras_to_check = 5
    
    progress_bar = st.progress(0)
    
    for i in range(max_cameras_to_check):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                available_cameras.append(i)
                
                # Show camera details
                col1, col2 = st.columns([1, 2])
                with col1:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(frame_rgb, caption=f"Camera {i}", use_column_width=True)
                
                with col2:
                    st.success(f"📷 Safety Camera {i} Detected")
                    st.write(f"**Resolution:** {frame.shape[1]}x{frame.shape[0]}")
                    st.write(f"**Channels:** {frame.shape[2]}")
                    st.write(f"**Status:** ✅ Operational")
            
            cap.release()
        
        progress_bar.progress((i + 1) / max_cameras_to_check)
    
    if available_cameras:
        st.success(f"🎯 Found {len(available_cameras)} operational camera(s) for safety monitoring")
    else:
        st.error("❌ No safety cameras detected. Please check connections and drivers.")

if __name__ == "__main__":
    main()
