import streamlit as st
import pandas as pd
import numpy as np
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

# Try to import OpenCV with error handling
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ OpenCV import error: {e}")
    CV2_AVAILABLE = False

# Try to import Ultralytics with error handling  
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ YOLO import error: {e}")
    YOLO_AVAILABLE = False

# Set page configuration with SafetyEagle branding
st.set_page_config(
    page_title="SafetyEagle AI - Oil & Gas PPE Monitoring",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Theme CSS for Oil & Gas SafetyEagle branding
st.markdown("""
<style>
    /* Main dark theme */
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0c0f15 0%, #1a1d25 100%);
    }
    
    /* Headers */
    .eagle-header {
        font-size: 3rem;
        background: linear-gradient(135deg, #FF6B35 0%, #FF8E53 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 800;
        margin-bottom: 0.5rem;
        font-family: 'Montserrat', sans-serif;
        text-shadow: 0 2px 10px rgba(255, 107, 53, 0.3);
    }
    
    .eagle-tagline {
        text-align: center;
        color: #94A3B8;
        font-size: 1.3rem;
        font-style: italic;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    .section-header {
        font-size: 1.8rem;
        color: #FF6B35;
        border-bottom: 3px solid #FF6B35;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(255, 107, 53, 0.2);
    }
    
    /* Cards and containers */
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #334155 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #FF6B35;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        border: 1px solid #334155;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #451a03 0%, #7c2d12 100%);
        border-left: 5px solid #F59E0B;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        border: 1px solid #7c2d12;
    }
    
    .success-card {
        background: linear-gradient(135deg, #064e3b 0%, #047857 100%);
        border-left: 5px solid #10B981;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        border: 1px solid #047857;
    }
    
    .project-info-card {
        background: linear-gradient(135deg, #7C3AED 0%, #A78BFA 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(124, 58, 237, 0.3);
        border: 1px solid #A78BFA;
    }
    
    .tab-container {
        background: linear-gradient(135deg, #1E293B 0%, #334155 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-top: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        border: 1px solid #334155;
    }
    
    .ppe-item {
        background: linear-gradient(135deg, #1E293B 0%, #2D3748 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem;
        border-left: 4px solid #FF6B35;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        border: 1px solid #2D3748;
    }
    
    /* Custom streamlit components */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        white-space: pre-wrap;
        background-color: #1E293B;
        border-radius: 8px 8px 0px 0px;
        gap: 8px;
        padding-top: 12px;
        padding-bottom: 12px;
        font-weight: 600;
        border: 1px solid #334155;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #FF6B35 !important;
        color: white !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1lcbmhc {
        background-color: #0F172A !important;
    }
    
    /* Input fields */
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select {
        background-color: #1E293B !important;
        color: #FAFAFA !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }
    
    .stSlider>div>div>div>div {
        background-color: #FF6B35 !important;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #FF6B35 0%, #FF8E53 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
        box-shadow: 0 2px 8px rgba(255, 107, 53, 0.3) !important;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #FF8E53 0%, #FF6B35 100%) !important;
        box-shadow: 0 4px 12px rgba(255, 107, 53, 0.4) !important;
        transform: translateY(-1px);
    }
    
    /* Dataframes and tables */
    .dataframe {
        background-color: #1E293B !important;
        color: #FAFAFA !important;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #1E293B !important;
        color: #FAFAFA !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }
    
    /* Plotly chart background */
    .js-plotly-plot .plotly, .modebar {
        background-color: #1E293B !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1E293B;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #FF6B35;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #FF8E53;
    }
    
    /* Status indicators */
    .status-active {
        color: #10B981;
        font-weight: 700;
    }
    
    .status-inactive {
        color: #EF4444;
        font-weight: 700;
    }
    
    /* Code blocks */
    .stCodeBlock {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with safe default values
def initialize_session_state():
    """Initialize all session state variables with safe defaults"""
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
            'confidence': 0.5,
            'speed': 'medium',
            'frame_skip': 3
        }
    if 'camera_urls' not in st.session_state:
        st.session_state.camera_urls = []
    if 'available_classes' not in st.session_state:
        st.session_state.available_classes = {}
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'demo_mode' not in st.session_state:
        st.session_state.demo_mode = False
    if 'project_info' not in st.session_state:
        st.session_state.project_info = {
            'company_name': 'ABC Oil & Gas Corp',
            'project_name': 'North Field Drilling Project',
            'engineer_name': 'John Safety Officer',
            'contractor_name': 'XYZ Contracting Ltd',
            'work_type': 'Drilling Operations',
            'project_hours': 240,
            'workers_assigned': 12,
            'start_date': datetime.now()
        }

# Standard Oil & Gas PPE classes
OIL_GAS_PPE_CLASSES = {
    0: "🛡️ Hard Hat",
    1: "👓 Safety Glasses", 
    2: "🦺 High-Vis Vest",
    3: "🧤 Safety Gloves",
    4: "👢 Safety Boots",
    5: "🎧 Hearing Protection",
    6: "🥽 Face Shield",
    7: "😷 Respirator",
    8: "🔥 Fire Retardant Clothing",
    9: "🪢 Harness",
    10: "📊 Gas Monitor"
}

def load_model_and_classes():
    """Load model with caching"""
    try:
        if not YOLO_AVAILABLE:
            st.error("❌ YOLO not available. Using demo mode.")
            return None, OIL_GAS_PPE_CLASSES, "demo"
            
        # Try multiple possible model paths
        possible_paths = [
            "models/best.pt",
            "best.pt", 
            "./models/best.pt",
            "model/best.pt"
        ]
        
        model = None
        loaded_path = None
        
        for model_path in possible_paths:
            try:
                if os.path.exists(model_path):
                    model = YOLO(model_path)
                    loaded_path = model_path
                    st.success(f"✅ Model loaded from: {model_path}")
                    break
            except Exception as e:
                st.warning(f"⚠️ Could not load from {model_path}: {e}")
                continue
        
        if model is None:
            # Fallback to standard PPE classes for demo
            st.warning("⚠️ Local model not found. Using Oil & Gas PPE classes for demo...")
            available_classes = OIL_GAS_PPE_CLASSES
            st.session_state.demo_mode = True
            return None, available_classes, "demo"
        
        # Get all available classes from the model
        if model and hasattr(model, 'names'):
            available_classes = model.names
            st.session_state.available_classes = available_classes
            st.session_state.model_loaded = True
            return model, available_classes, loaded_path
        else:
            st.error("❌ Could not extract class names from model")
            return None, OIL_GAS_PPE_CLASSES, "demo"
            
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.info("🦅 Switching to SafetyEagle Demo Mode...")
        return None, OIL_GAS_PPE_CLASSES, "demo"

def initialize_app():
    """Initialize the app and load model on startup"""
    initialize_session_state()
    
    if not st.session_state.model_loaded:
        with st.spinner("🦅 Initializing SafetyEagle AI System for Oil & Gas Safety..."):
            model, available_classes, model_path = load_model_and_classes()
            st.session_state.model = model
            st.session_state.available_classes = available_classes
            st.session_state.model_loaded = True

def main():
    # SafetyEagle Header with Oil & Gas Theme
    st.markdown('<h1 class="eagle-header">🦅 SafetyEagle AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="eagle-tagline">Advanced PPE Monitoring for Oil & Gas Industry</p>', unsafe_allow_html=True)
    
    # Initialize app on startup
    initialize_app()
    
    # Show system status in sidebar with dark theme
    st.sidebar.markdown("### 🦅 SafetyEagle Status")
    st.sidebar.markdown("---")
    
    # Display project info if available
    if st.session_state.project_info['project_name']:
        st.sidebar.markdown("### 📋 Project Info")
        st.sidebar.markdown(f"**Project:** {st.session_state.project_info['project_name']}")
        st.sidebar.markdown(f"**Company:** {st.session_state.project_info['company_name']}")
        st.sidebar.markdown(f"**Engineer:** {st.session_state.project_info['engineer_name']}")
        st.sidebar.markdown("---")
    
    # System status with colored indicators
    status_col1, status_col2 = st.sidebar.columns(2)
    with status_col1:
        if CV2_AVAILABLE:
            st.markdown('<p class="status-active">✅ OpenCV</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-inactive">❌ OpenCV</p>', unsafe_allow_html=True)
            
    with status_col2:
        if YOLO_AVAILABLE:
            st.markdown('<p class="status-active">✅ YOLO</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-inactive">❌ YOLO</p>', unsafe_allow_html=True)
    
    if st.session_state.model_loaded and st.session_state.available_classes:
        if st.session_state.demo_mode:
            st.sidebar.markdown('<div class="warning-card">🟡 Demo Mode Active</div>', unsafe_allow_html=True)
        else:
            st.sidebar.markdown('<div class="success-card">✅ Model Loaded</div>', unsafe_allow_html=True)
        st.sidebar.info(f"**Available Classes:** {len(st.session_state.available_classes)}")
    else:
        st.sidebar.markdown('<div class="warning-card">❌ Model Not Loaded</div>', unsafe_allow_html=True)
        st.sidebar.info("Using simulation mode")
    
    # Quick actions in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚡ Quick Actions")
    if st.sidebar.button("🔄 Reset App", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    if st.sidebar.button("📊 Demo Data", use_container_width=True):
        generate_demo_data()
        st.sidebar.success("Demo data loaded!")
    
    # Main tabs for workflow
    tabs = st.tabs([
        "🏢 Project Setup", 
        "🛡️ PPE Selection", 
        "📷 Camera Setup", 
        "⚙️ Detection Settings", 
        "📊 Dashboard", 
        "📈 Reports"
    ])
    
    with tabs[0]:
        show_project_setup()
    with tabs[1]:
        show_ppe_selection()
    with tabs[2]:
        show_camera_setup()
    with tabs[3]:
        show_detection_settings()
    with tabs[4]:
        show_dashboard()
    with tabs[5]:
        show_reports()

def show_project_setup():
    """Project setup tab"""
    st.markdown('<h2 class="section-header">🏢 Project Setup</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="tab-container">', unsafe_allow_html=True)
    st.info("Configure your oil & gas project details for comprehensive safety monitoring")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏭 Company Information")
        company_name = st.text_input("Company Name", value=st.session_state.project_info['company_name'])
        project_name = st.text_input("Project Name", value=st.session_state.project_info['project_name'])
        engineer_name = st.text_input("Safety Engineer Name", value=st.session_state.project_info['engineer_name'])
        contractor_name = st.text_input("Contractor Name (if applicable)", value=st.session_state.project_info['contractor_name'])
    
    with col2:
        st.subheader("🔧 Project Details")
        work_type = st.selectbox(
            "Type of Work",
            ["Drilling Operations", "Well Maintenance", "Pipeline Construction", 
             "Refinery Maintenance", "Offshore Operations", "Hazardous Area Work", "Other"],
            index=0
        )
        
        project_hours = st.number_input(
            "Project Duration (hours)", 
            min_value=1, 
            max_value=10000,
            value=max(1, st.session_state.project_info['project_hours']),
            step=1
        )
        
        workers_assigned = st.number_input(
            "Number of Workers Assigned", 
            min_value=1, 
            max_value=500,
            value=max(1, st.session_state.project_info['workers_assigned']),
            step=1
        )
        
        start_date = st.date_input("Project Start Date", value=st.session_state.project_info['start_date'])
    
    # Save project information
    if st.button("💾 Save Project Configuration", type="primary", use_container_width=True):
        st.session_state.project_info = {
            'company_name': company_name,
            'project_name': project_name,
            'engineer_name': engineer_name,
            'contractor_name': contractor_name,
            'work_type': work_type,
            'project_hours': project_hours,
            'workers_assigned': workers_assigned,
            'start_date': start_date
        }
        st.success("✅ Project configuration saved successfully!")
        
        # Display project summary
        st.subheader("📋 Project Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **Company:** {company_name}  
            **Project:** {project_name}  
            **Safety Engineer:** {engineer_name}  
            **Contractor:** {contractor_name if contractor_name else "N/A"}
            """)
        with col2:
            st.markdown(f"""
            **Work Type:** {work_type}  
            **Duration:** {project_hours} hours  
            **Workers:** {workers_assigned}  
            **Start Date:** {start_date}
            """)
    st.markdown('</div>', unsafe_allow_html=True)

def show_ppe_selection():
    """PPE selection tab"""
    st.markdown('<h2 class="section-header">🛡️ PPE Equipment Selection</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="tab-container">', unsafe_allow_html=True)
    
    # Use demo classes if model not loaded
    if not st.session_state.model_loaded or not st.session_state.available_classes:
        st.session_state.available_classes = OIL_GAS_PPE_CLASSES
        st.info("🦅 Using standard Oil & Gas PPE classes for demonstration")
    
    st.info("Select the required Personal Protective Equipment for your oil & gas project")
    
    # Display available PPE classes
    available_classes = st.session_state.available_classes
    
    st.subheader("🛠️ Available Safety Equipment")
    
    # Create columns for better organization
    num_columns = 3
    classes_list = list(available_classes.items())
    
    if classes_list:
        classes_per_column = (len(classes_list) + num_columns - 1) // num_columns
        cols = st.columns(num_columns)
        
        selected_classes = st.session_state.selected_ppe.copy()
        
        for i, (class_id, class_name) in enumerate(classes_list):
            col_idx = i // classes_per_column
            with cols[col_idx]:
                is_selected = st.checkbox(
                    f"**{class_name}**",
                    value=class_id in selected_classes,
                    key=f"ppe_{class_id}",
                    help=f"Class ID: {class_id}"
                )
                if is_selected:
                    selected_classes[class_id] = class_name
                elif class_id in selected_classes:
                    del selected_classes[class_id]
        
        # Selection actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("✅ Select All PPE", use_container_width=True):
                st.session_state.selected_ppe = available_classes.copy()
                st.success("✅ All PPE equipment selected!")
                st.rerun()
        
        with col2:
            if st.button("🔄 Basic Oil & Gas Set", use_container_width=True):
                # Basic selection for oil & gas
                basic_ppe = {0: "🛡️ Hard Hat", 1: "👓 Safety Glasses", 2: "🦺 High-Vis Vest", 3: "🧤 Safety Gloves", 4: "👢 Safety Boots"}
                st.session_state.selected_ppe = {k: v for k, v in available_classes.items() if v in basic_ppe.values()}
                st.info("🔄 Basic oil & gas PPE set selected")
                st.rerun()
        
        with col3:
            if st.button("💾 Save PPE Selection", type="primary", use_container_width=True):
                st.session_state.selected_ppe = selected_classes
                st.success(f"✅ Saved {len(selected_classes)} PPE items for monitoring!")
        
        # Show current selection
        if selected_classes:
            st.subheader("🎯 Current PPE Selection")
            st.info(f"**Selected {len(selected_classes)} out of {len(available_classes)} PPE items:**")
            
            # Display in a nice grid
            selected_items = list(selected_classes.values())
            num_cols = 3
            summary_cols = st.columns(num_cols)
            
            for i, item in enumerate(selected_items):
                with summary_cols[i % num_cols]:
                    st.markdown(f'<div class="ppe-item">✅ <strong>{item}</strong></div>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ No PPE selected. Please select at least one safety equipment item.")
    else:
        st.error("❌ No PPE classes available. Please check model initialization.")
    st.markdown('</div>', unsafe_allow_html=True)

def show_camera_setup():
    """Camera setup tab"""
    st.markdown('<h2 class="section-header">📷 Camera Configuration</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="tab-container">', unsafe_allow_html=True)
    st.info("Configure camera sources for real-time safety monitoring")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🌐 IP Camera Setup")
        camera_url = st.text_input(
            "IP Camera URL:",
            placeholder="rtsp://username:password@ip:port/stream or http://ip:port/video",
            help="Enter your IP camera stream URL"
        )
    
    with col2:
        st.write("")
        st.write("")
        if st.button("🔗 Test Connection", use_container_width=True):
            if camera_url:
                if CV2_AVAILABLE:
                    try:
                        cap = cv2.VideoCapture(camera_url)
                        if cap.isOpened():
                            ret, frame = cap.read()
                            cap.release()
                            if ret:
                                st.success("✅ Camera connected successfully!")
                                if camera_url not in st.session_state.camera_urls:
                                    st.session_state.camera_urls.append(camera_url)
                            else:
                                st.error("❌ Connected but no frame received")
                        else:
                            st.error("❌ Cannot connect to camera")
                    except Exception as e:
                        st.error(f"❌ Connection test failed: {e}")
                else:
                    st.error("❌ OpenCV not available for camera testing")
            else:
                st.error("Please enter a camera URL")
    
    # Camera management
    if st.session_state.camera_urls:
        st.subheader("💾 Saved Cameras")
        for i, url in enumerate(st.session_state.camera_urls):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.code(url)
            with col2:
                if st.button("🗑️", key=f"remove_{i}"):
                    st.session_state.camera_urls.pop(i)
                    st.rerun()
    
    # Live monitoring section
    st.subheader("🎥 Live Monitoring")
    if st.session_state.camera_urls:
        selected_camera = st.selectbox("Select Camera", st.session_state.camera_urls)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start Monitoring", type="primary", use_container_width=True):
                st.session_state.monitoring = True
                st.success("🚀 Starting live monitoring...")
        with col2:
            if st.button("⏹️ Stop Monitoring", use_container_width=True):
                st.session_state.monitoring = False
                st.info("Monitoring stopped")
    else:
        st.warning("No cameras configured. Add a camera URL above.")
    st.markdown('</div>', unsafe_allow_html=True)

def show_detection_settings():
    """Detection settings tab"""
    st.markdown('<h2 class="section-header">⚙️ Detection Settings</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="tab-container">', unsafe_allow_html=True)
    st.info("Configure AI detection parameters for optimal safety monitoring")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🎯 Detection Sensitivity")
        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=st.session_state.detection_settings.get('confidence', 0.5),
            step=0.05,
            help="Higher values = fewer but more accurate detections"
        )
    
    with col2:
        st.subheader("⚡ Processing Speed")
        speed_setting = st.selectbox(
            "Detection Speed",
            options=["fast", "medium", "accurate"],
            index=1,
            help="Balance between speed and accuracy"
        )
    
    with col3:
        st.subheader("📊 Frame Processing")
        frame_skip = st.slider(
            "Frame Skip Rate",
            min_value=1,
            max_value=10,
            value=st.session_state.detection_settings.get('frame_skip', 3),
            help="Process every Nth frame for performance"
        )
    
    # Save settings
    if st.button("💾 Save Detection Settings", type="primary", use_container_width=True):
        st.session_state.detection_settings = {
            'confidence': confidence,
            'speed': speed_setting,
            'frame_skip': frame_skip
        }
        st.success("✅ Detection settings saved successfully!")
    
    # Current configuration display
    st.subheader("🔧 Current Configuration")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🏢 Project Info</h4>
            <p><strong>Project:</strong> {st.session_state.project_info['project_name'] or 'Not set'}</p>
            <p><strong>Work Type:</strong> {st.session_state.project_info['work_type']}</p>
            <p><strong>Workers:</strong> {st.session_state.project_info['workers_assigned']}</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🛡️ Safety Setup</h4>
            <p><strong>PPE Items:</strong> {len(st.session_state.selected_ppe)}</p>
            <p><strong>Confidence:</strong> {confidence}</p>
            <p><strong>Speed:</strong> {speed_setting.title()}</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def show_dashboard():
    """Enhanced dashboard tab"""
    st.markdown('<h2 class="section-header">📊 Safety Dashboard</h2>', unsafe_allow_html=True)
    
    # Project info header
    if st.session_state.project_info['project_name']:
        st.markdown(f"""
        <div class="project-info-card">
            <h3>🏢 {st.session_state.project_info['project_name']}</h3>
            <p>{st.session_state.project_info['company_name']} • {st.session_state.project_info['work_type']} • {st.session_state.project_info['workers_assigned']} Workers</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="tab-container">', unsafe_allow_html=True)
    
    if not st.session_state.violations:
        st.info("📊 No safety violations recorded yet. Start monitoring to see analytics.")
        
        # Demo data for visualization
        if st.button("🧪 Load Demo Data for Visualization", use_container_width=True):
            generate_demo_data()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Key metrics
    st.subheader("📈 Key Safety Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_violations = len(st.session_state.violations)
        st.markdown(f'<div class="metric-card"><h3>{total_violations}</h3><p>Total Violations</p></div>', unsafe_allow_html=True)
    
    with col2:
        today_violations = len([v for v in st.session_state.violations 
                              if v['timestamp'].date() == datetime.now().date()])
        st.markdown(f'<div class="metric-card"><h3>{today_violations}</h3><p>Today\'s Violations</p></div>', unsafe_allow_html=True)
    
    with col3:
        workers = max(1, st.session_state.project_info['workers_assigned'])
        compliance_rate = max(0, 100 - (total_violations / workers * 10))
        st.markdown(f'<div class="metric-card"><h3>{compliance_rate:.1f}%</h3><p>Compliance Rate</p></div>', unsafe_allow_html=True)
    
    with col4:
        avg_violations_per_worker = total_violations / workers
        st.markdown(f'<div class="metric-card"><h3>{avg_violations_per_worker:.1f}</h3><p>Avg Violations/Worker</p></div>', unsafe_allow_html=True)
    
    # Charts and visualizations
    st.subheader("📊 Safety Analytics")
    
    # User-selectable chart types
    chart_type = st.selectbox(
        "Select Chart Type",
        ["Violations Over Time", "PPE Compliance by Equipment", "Worker vs Violations", "Heatmap Analysis"],
        help="Choose the type of safety analysis to display"
    )
    
    if chart_type == "Violations Over Time":
        show_violations_over_time()
    elif chart_type == "PPE Compliance by Equipment":
        show_ppe_compliance()
    elif chart_type == "Worker vs Violations":
        show_worker_violations_correlation()
    elif chart_type == "Heatmap Analysis":
        show_heatmap_analysis()
    
    # Recent violations
    st.subheader("🚨 Recent Safety Violations")
    for i, violation in enumerate(st.session_state.violations[-5:]):
        with st.expander(f"Violation {i+1} - {violation['timestamp'].strftime('%H:%M:%S')}"):
            col1, col2 = st.columns([1, 2])
            with col1:
                # Create a simple placeholder for demo
                img_placeholder = np.ones((200, 300, 3), dtype=np.uint8) * 100
                cv2.putText(img_placeholder, "Violation Capture", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                st.image(img_placeholder, use_column_width=True)
            with col2:
                st.write(f"**Missing PPE:** {violation['missing_classes']}")
                st.write(f"**Time:** {violation['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"**Work Type:** {st.session_state.project_info['work_type']}")
    st.markdown('</div>', unsafe_allow_html=True)

def show_violations_over_time():
    """Show violations over time chart with dark theme"""
    df = pd.DataFrame([
        {
            'timestamp': v['timestamp'],
            'hour': v['timestamp'].hour,
            'date': v['timestamp'].date()
        }
        for v in st.session_state.violations
    ])
    
    if not df.empty:
        # Daily violations
        daily_data = df.groupby('date').size().reset_index(name='violations')
        fig = px.line(daily_data, x='date', y='violations', 
                     title="Safety Violations Trend - Daily",
                     markers=True)
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            title_font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Hourly distribution
        hourly_data = df.groupby('hour').size().reset_index(name='violations')
        fig2 = px.bar(hourly_data, x='hour', y='violations',
                     title="Violations by Hour of Day",
                     color='violations')
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            title_font_color='white'
        )
        st.plotly_chart(fig2, use_container_width=True)

def show_ppe_compliance():
    """Show PPE compliance by equipment type with dark theme"""
    equipment_violations = {}
    for violation in st.session_state.violations:
        if isinstance(violation['missing_classes'], str):
            missing_items = violation['missing_classes'].split(', ')
        else:
            missing_items = [violation['missing_classes']]
        for item in missing_items:
            equipment_violations[item] = equipment_violations.get(item, 0) + 1
    
    if equipment_violations:
        fig = px.pie(values=list(equipment_violations.values()), 
                    names=list(equipment_violations.keys()),
                    title="PPE Violations Distribution")
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            title_font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)

def show_worker_violations_correlation():
    """Show correlation between workers and violations with dark theme"""
    workers = max(1, st.session_state.project_info['workers_assigned'])
    violations = len(st.session_state.violations)
    
    # Simulate worker-wise data for demo
    worker_data = []
    for i in range(workers):
        worker_violations = np.random.poisson(max(1, violations / workers))
        worker_data.append({
            'Worker_ID': f"W{i+1:03d}",
            'Violations': worker_violations,
            'Department': np.random.choice(['Drilling', 'Maintenance', 'Operations', 'Safety'])
        })
    
    df_workers = pd.DataFrame(worker_data)
    
    fig = px.scatter(df_workers, x='Worker_ID', y='Violations', color='Department',
                    title="Worker Safety Performance",
                    size='Violations', hover_data=['Department'])
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        title_font_color='white'
    )
    st.plotly_chart(fig, use_container_width=True)

def show_heatmap_analysis():
    """Show heatmap analysis with dark theme"""
    # Generate time-based heatmap data
    hours = list(range(24))
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    # Simulate heatmap data for demo
    heatmap_data = np.random.poisson(1, (7, 24))
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=hours,
        y=days,
        colorscale='Viridis',
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Safety Violations Heatmap (Time vs Day)",
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        title_font_color='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_reports():
    """Professional reports tab"""
    st.markdown('<h2 class="section-header">📈 Professional Safety Reports</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="tab-container">', unsafe_allow_html=True)
    
    if not st.session_state.project_info['project_name']:
        st.warning("⚠️ Please complete Project Setup first to generate professional reports.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    st.info("Generate comprehensive safety reports for management and regulatory compliance")
    
    # Report type selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📋 Quick Report")
        if st.button("📊 Generate Summary Report", use_container_width=True):
            generate_summary_report()
    
    with col2:
        st.subheader("📑 Detailed Analysis")
        if st.button("📈 Comprehensive Report", use_container_width=True):
            generate_comprehensive_report()
    
    with col3:
        st.subheader("🔄 Regulatory Compliance")
        if st.button("⚖️ Compliance Report", use_container_width=True):
            generate_compliance_report()
    
    # Custom report configuration
    st.subheader("🎛️ Custom Report Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        report_period = st.selectbox(
            "Report Period",
            ["Last 7 Days", "Last 30 Days", "Project to Date", "Custom Range"]
        )
        include_charts = st.checkbox("Include Charts and Graphs", value=True)
        include_recommendations = st.checkbox("Include Safety Recommendations", value=True)
    
    with col2:
        report_format = st.selectbox(
            "Report Format",
            ["PDF", "Excel", "HTML", "Word"]
        )
        data_detail = st.select_slider(
            "Data Detail Level",
            options=["Summary", "Standard", "Detailed", "Comprehensive"]
        )
    
    if st.button("🚀 Generate Custom Report", type="primary", use_container_width=True):
        generate_custom_report(report_period, report_format, data_detail, include_charts, include_recommendations)
    
    # Recent report history
    if st.session_state.violations:
        st.subheader("📁 Recent Reports")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                "📥 Download CSV Data",
                pd.DataFrame(st.session_state.violations).to_csv(index=False),
                "safety_data.csv",
                "text/csv"
            )
        
        with col2:
            st.download_button(
                "📊 Download Excel Report",
                generate_excel_report(),
                "safety_report.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col3:
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ... (Keep all the existing helper functions: generate_summary_report, generate_comprehensive_report, etc.)
# The helper functions remain the same as in the previous version

def generate_summary_report():
    """Generate summary safety report"""
    st.success("📋 Generating Summary Safety Report...")
    
    # Create report content
    report_content = f"""
    # SafetyEagle AI - Safety Summary Report
    ## {st.session_state.project_info['company_name']}
    ### Project: {st.session_state.project_info['project_name']}
    
    **Report Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
    **Safety Engineer:** {st.session_state.project_info['engineer_name']}
    
    ## Executive Summary
    - Total Workers: {st.session_state.project_info['workers_assigned']}
    - Total Violations: {len(st.session_state.violations)}
    - Overall Compliance Rate: {calculate_compliance_rate():.1f}%
    
    ## Key Findings
    - Most common violation: {get_most_common_violation()}
    - Peak violation hours: {get_peak_violation_hours()}
    - Recommended actions: {get_safety_recommendations()}
    
    ## Safety Performance
    {generate_performance_metrics()}
    """
    
    st.download_button(
        "📥 Download Summary Report",
        report_content,
        f"Safety_Summary_{datetime.now().strftime('%Y%m%d')}.txt",
        "text/plain"
    )

def generate_comprehensive_report():
    """Generate comprehensive safety report"""
    st.success("📈 Generating Comprehensive Safety Report...")
    
    # This would typically generate Excel/PDF with charts
    st.info("Comprehensive report generation would include:")
    st.markdown("""
    - Detailed violation analytics
    - Department-wise performance
    - Trend analysis charts
    - Compliance scoring
    - Action plan recommendations
    - Regulatory compliance checklist
    """)

def generate_compliance_report():
    """Generate regulatory compliance report"""
    st.success("⚖️ Generating Regulatory Compliance Report...")
    
    st.info("Compliance report includes:")
    st.markdown("""
    - OSHA compliance checklist
    - Industry standards adherence
    - Audit trail documentation
    - Corrective action records
    - Training compliance status
    - Equipment certification tracking
    """)

def generate_custom_report(period, format, detail, charts, recommendations):
    """Generate custom report based on user preferences"""
    st.success(f"🚀 Generating Custom {format} Report...")
    st.write(f"**Configuration:** {period}, {detail} detail, Charts: {charts}, Recommendations: {recommendations}")
    
    # Simulate report generation
    time.sleep(2)
    st.balloons()
    st.success(f"✅ Custom {format} report generated successfully!")

def generate_excel_report():
    """Generate Excel report with safety data"""
    # Create a simple Excel file in memory
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Summary sheet
        summary_data = {
            'Metric': ['Total Violations', 'Workers', 'Compliance Rate', 'Project Hours'],
            'Value': [
                len(st.session_state.violations),
                st.session_state.project_info['workers_assigned'],
                f"{calculate_compliance_rate():.1f}%",
                st.session_state.project_info['project_hours']
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        # Violations sheet
        if st.session_state.violations:
            violations_df = pd.DataFrame(st.session_state.violations)
            violations_df.to_excel(writer, sheet_name='Violations', index=False)
    
    return output.getvalue()

def calculate_compliance_rate():
    """Calculate overall compliance rate"""
    workers = max(1, st.session_state.project_info['workers_assigned'])
    violations = len(st.session_state.violations)
    return max(0, 100 - (violations / workers * 10))

def get_most_common_violation():
    """Get most common violation type"""
    if not st.session_state.violations:
        return "No violations recorded"
    
    violations_count = {}
    for violation in st.session_state.violations:
        if isinstance(violation['missing_classes'], str):
            items = violation['missing_classes'].split(', ')
        else:
            items = [violation['missing_classes']]
        for item in items:
            violations_count[item] = violations_count.get(item, 0) + 1
    
    return max(violations_count, key=violations_count.get) if violations_count else "N/A"

def get_peak_violation_hours():
    """Get peak violation hours"""
    if not st.session_state.violations:
        return "No data"
    
    hours = [v['timestamp'].hour for v in st.session_state.violations]
    if not hours:
        return "N/A"
    peak_hour = max(set(hours), key=hours.count)
    return f"{peak_hour}:00 - {peak_hour+1}:00"

def get_safety_recommendations():
    """Generate safety recommendations"""
    if not st.session_state.violations:
        return "Continue current safety protocols"
    
    total_violations = len(st.session_state.violations)
    workers = max(1, st.session_state.project_info['workers_assigned'])
    
    if total_violations / workers > 0.5:
        return "Immediate safety training required"
    elif total_violations / workers > 0.2:
        return "Enhanced monitoring and refresher training recommended"
    else:
        return "Maintain current safety standards"

def generate_performance_metrics():
    """Generate performance metrics string"""
    workers = max(1, st.session_state.project_info['workers_assigned'])
    violations = len(st.session_state.violations)
    
    metrics = f"""
    - Safety Compliance Score: {calculate_compliance_rate():.1f}%
    - Violations per Worker: {violations / workers:.2f}
    - Daily Average Violations: {violations / max(1, (datetime.now().date() - st.session_state.project_info['start_date']).days):.1f}
    - Project Safety Rating: {'⭐' * min(5, max(1, 6 - violations // workers))}
    """
    return metrics

def generate_demo_data():
    """Generate demo data for visualization"""
    demo_violations = []
    for i in range(50):
        demo_violations.append({
            'timestamp': datetime.now() - pd.Timedelta(hours=np.random.randint(1, 168)),
            'missing_classes': ', '.join(np.random.choice(list(st.session_state.selected_ppe.values()), 
                                              np.random.randint(1, 3))),
            'selected_classes': list(st.session_state.selected_ppe.values())
        })
    st.session_state.violations = demo_violations

if __name__ == "__main__":
    main()
