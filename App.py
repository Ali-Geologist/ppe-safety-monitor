import streamlit as st
try:
    import cv2
except ImportError:
    st.error("OpenCV not installed properly")
import pandas as pd
import numpy as np
from ultralytics import YOLO
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from pathlib import Path
import tempfile
from PIL import Image
import plotly.express as px
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
    page_title="SafetyEagle AI - PPE Monitoring",
    page_icon="🦅",
    layout="wide"
)

# Custom CSS for SafetyEagle branding
st.markdown("""
<style>
    .eagle-header {
        font-size: 2.8rem;
        color: #8B4513;
        background: linear-gradient(135deg, #8B4513 0%, #FF6700 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.5rem;
        font-family: 'Montserrat', sans-serif;
    }
    .eagle-tagline {
        text-align: center;
        color: #2F4F4F;
        font-size: 1.2rem;
        font-style: italic;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #8B4513;
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
    .section-header {
        font-size: 1.5rem;
        color: #8B4513;
        border-bottom: 2px solid #8B4513;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
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

# Load model with caching and get all classes
@st.cache_resource(show_spinner="Loading SafetyEagle AI Model...")
def load_model_and_classes():
    try:
        if not YOLO_AVAILABLE:
            st.error("❌ YOLO not available. Using demo mode.")
            return None, {}, "demo"
            
        # Try multiple possible model paths for Streamlit Cloud
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
            # Fallback to nano model for demo
            st.warning("⚠️ Local model not found. Using YOLOv8n for demo...")
            model = YOLO('yolov8n.pt')
            loaded_path = 'yolov8n.pt'
            st.session_state.demo_mode = True
        
        # Get all available classes from the model
        if model and hasattr(model, 'names'):
            available_classes = model.names
            st.session_state.available_classes = available_classes
            st.session_state.model_loaded = True
            return model, available_classes, loaded_path
        else:
            st.error("❌ Could not extract class names from model")
            return None, {}, loaded_path
            
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.info("🦅 Switching to SafetyEagle Demo Mode...")
        return None, {}, "demo"

def initialize_app():
    """Initialize the app and load model on startup"""
    if not st.session_state.model_loaded:
        with st.spinner("🦅 Initializing SafetyEagle AI System..."):
            model, available_classes, model_path = load_model_and_classes()
            st.session_state.model = model
            st.session_state.available_classes = available_classes
            st.session_state.model_loaded = True
            
            # Auto-select all classes by default
            if available_classes:
                st.session_state.selected_ppe = available_classes.copy()
                st.success(f"✅ Loaded {len(available_classes)} detection classes")

def validate_ip_camera_url(url):
    """Validate IP camera URL"""
    if not url:
        return False, "URL cannot be empty"
    
    # Basic URL validation
    ip_pattern = r'^rtsp://|^http://|^https://'
    if not re.match(ip_pattern, url):
        return False, "URL must start with rtsp://, http://, or https://"
    
    return True, "URL looks valid"

def test_ip_camera(url, timeout=5):
    """Test if IP camera URL is accessible"""
    if not CV2_AVAILABLE:
        return False, "OpenCV not available for camera testing"
        
    try:
        if url.startswith('rtsp://'):
            # Test RTSP stream
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    return True, "✅ RTSP camera connected successfully!"
                else:
                    return False, "❌ RTSP camera connected but no frame received"
            else:
                return False, "❌ Cannot connect to RTSP stream"
        
        elif url.startswith(('http://', 'https://')):
            # Test HTTP stream
            try:
                response = requests.get(url, timeout=timeout, stream=True)
                if response.status_code == 200:
                    return True, "✅ HTTP camera connected successfully!"
                else:
                    return False, f"❌ HTTP camera returned status code: {response.status_code}"
            except requests.exceptions.RequestException as e:
                return False, f"❌ HTTP camera connection failed: {e}"
        
        else:
            return False, "❌ Unsupported URL protocol"
    
    except Exception as e:
        return False, f"❌ Camera test failed: {e}"

def main():
    # SafetyEagle Header
    st.markdown('<h1 class="eagle-header">🦅 SafetyEagle AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="eagle-tagline">Soaring Above Safety Standards</p>', unsafe_allow_html=True)
    
    # Initialize app on startup
    initialize_app()
    
    # Show system status in sidebar
    st.sidebar.markdown("### 🦅 SafetyEagle Status")
    st.sidebar.markdown("---")
    
    if not CV2_AVAILABLE:
        st.sidebar.error("❌ OpenCV Not Available")
        st.sidebar.info("Using limited functionality mode")
    
    if not YOLO_AVAILABLE:
        st.sidebar.error("❌ YOLO Not Available")
        st.sidebar.info("Using demo simulation mode")
    
    if st.session_state.model_loaded and st.session_state.available_classes:
        if st.session_state.demo_mode:
            st.sidebar.warning("🟡 Demo Mode Active")
        else:
            st.sidebar.success("✅ Model Loaded")
        st.sidebar.info(f"**Available Classes:** {len(st.session_state.available_classes)}")
        
        # Show quick class overview
        with st.sidebar.expander("📋 Quick Class Overview"):
            for class_id, class_name in list(st.session_state.available_classes.items())[:8]:
                status = "✅" if class_id in st.session_state.selected_ppe else "❌"
                st.write(f"{status} **{class_id}:** {class_name}")
            if len(st.session_state.available_classes) > 8:
                st.write(f"... and {len(st.session_state.available_classes) - 8} more classes")
    else:
        st.sidebar.error("❌ Model Not Loaded")
        st.sidebar.info("Using simulation mode")
    
    # Sidebar navigation
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧭 Navigation")
    page = st.sidebar.radio("Go to", 
        ["Class Selection", "Camera Setup", "Settings", "Live Monitoring", "Dashboard", "Reports", "Deployment Guide"])
    
    if page == "Class Selection":
        show_class_selection()
    elif page == "Camera Setup":
        show_camera_setup()
    elif page == "Settings":
        show_settings()
    elif page == "Live Monitoring":
        show_live_monitoring()
    elif page == "Dashboard":
        show_dashboard()
    elif page == "Reports":
        show_reports()
    elif page == "Deployment Guide":
        show_deployment_guide()

def show_class_selection():
    st.markdown('<h2 class="section-header">🎯 Select Detection Classes</h2>', unsafe_allow_html=True)
    
    if not st.session_state.model_loaded:
        st.error("❌ Model not loaded. Please wait for model initialization.")
        if st.button("🔄 Retry Model Loading"):
            st.session_state.model_loaded = False
            st.rerun()
        return
    
    if not st.session_state.available_classes:
        st.error("❌ No classes available from the model.")
        return
    
    if st.session_state.demo_mode:
        st.warning("🦅 **SafetyEagle Demo Mode**: Using YOLOv8n pretrained model for demonstration.")
    
    st.success(f"🎉 **Model loaded successfully! Found {len(st.session_state.available_classes)} detection classes**")
    
    # Display all available classes for selection
    st.subheader("Available Detection Classes")
    st.info("Select which safety classes you want to monitor:")
    
    # Create columns for better organization
    num_columns = 3
    classes_list = list(st.session_state.available_classes.items())
    classes_per_column = (len(classes_list) + num_columns - 1) // num_columns
    
    cols = st.columns(num_columns)
    
    selected_classes = st.session_state.selected_ppe.copy()
    
    for i, (class_id, class_name) in enumerate(classes_list):
        col_idx = i // classes_per_column
        with cols[col_idx]:
            is_selected = st.checkbox(
                f"**Class {class_id}:** {class_name}",
                value=class_id in selected_classes,
                key=f"class_{class_id}"
            )
            if is_selected:
                selected_classes[class_id] = class_name
            elif class_id in selected_classes:
                del selected_classes[class_id]
    
    # Selection actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ Select All Classes", use_container_width=True):
            st.session_state.selected_ppe = st.session_state.available_classes.copy()
            st.success("✅ All classes selected!")
            st.rerun()
    
    with col2:
        if st.button("❌ Clear All Selections", use_container_width=True):
            st.session_state.selected_ppe = {}
            st.info("🗑️ All selections cleared")
            st.rerun()
    
    with col3:
        if st.button("💾 Save Selection", type="primary", use_container_width=True):
            st.session_state.selected_ppe = selected_classes
            st.success(f"✅ Saved {len(selected_classes)} classes for monitoring!")
    
    # Show current selection summary
    if selected_classes:
        st.subheader("Current Selection Summary")
        st.info(f"**Selected {len(selected_classes)} out of {len(st.session_state.available_classes)} classes:**")
        
        # Display selected classes in a nice format
        selected_items = list(selected_classes.items())
        num_cols = 4
        summary_cols = st.columns(num_cols)
        
        for i, (class_id, class_name) in enumerate(selected_items):
            with summary_cols[i % num_cols]:
                st.markdown(f"📍 **{class_id}:** {class_name}")
    else:
        st.warning("⚠️ No classes selected. Please select at least one class to enable monitoring.")

def show_camera_setup():
    st.markdown('<h2 class="section-header">📷 Camera Configuration</h2>', unsafe_allow_html=True)
    
    if not st.session_state.model_loaded:
        st.error("❌ Please wait for model initialization on the Class Selection page first.")
        return
    
    if not CV2_AVAILABLE:
        st.error("❌ OpenCV not available. Camera functionality limited.")
        st.info("🦅 SafetyEagle AI can still analyze uploaded videos and images.")
    
    st.subheader("1. IP Camera / Network Stream")
    st.info("Connect to IP cameras, RTSP streams, or network cameras")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        camera_url = st.text_input(
            "IP Camera URL:",
            placeholder="rtsp://username:password@ip:port/stream or http://ip:port/video"
        )
    
    with col2:
        st.write("")
        st.write("")
        if st.button("🔗 Test Connection"):
            if camera_url:
                is_valid, message = validate_ip_camera_url(camera_url)
                if is_valid:
                    with st.spinner("Testing camera connection..."):
                        success, result = test_ip_camera(camera_url)
                    if success:
                        st.success(result)
                        if camera_url not in st.session_state.camera_urls:
                            st.session_state.camera_urls.append(camera_url)
                            st.success("✅ Camera added to saved list!")
                    else:
                        st.error(result)
                else:
                    st.error(message)
            else:
                st.error("Please enter a camera URL")
    
    # Common camera URL examples
    with st.expander("📋 Common Camera URL Formats"):
        st.markdown("""
        **RTSP Examples:**
        - `rtsp://username:password@192.168.1.100:554/stream1`
        - `rtsp://admin:password@camera_ip:554/11`
        
        **HTTP Examples:**
        - `http://192.168.1.100:8080/video`
        - `http://192.168.1.100:4747/video`
        """)
    
    # Saved camera URLs
    if st.session_state.camera_urls:
        st.subheader("💾 Saved Camera URLs")
        for i, url in enumerate(st.session_state.camera_urls):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.code(url, language="text")
            with col2:
                if st.button("🔗 Test", key=f"test_{i}"):
                    with st.spinner("Testing..."):
                        success, result = test_ip_camera(url)
                    if success:
                        st.success(result)
                    else:
                        st.error(result)
            with col3:
                if st.button("🗑️ Remove", key=f"remove_{i}"):
                    st.session_state.camera_urls.pop(i)
                    st.rerun()
    
    st.subheader("2. Video File Upload")
    st.info("Use pre-recorded videos for safety analysis")
    
    uploaded_file = st.file_uploader("Upload video file", type=['mp4', 'avi', 'mov', 'mkv'])
    if uploaded_file:
        st.success(f"✅ Video uploaded: {uploaded_file.name}")

def show_settings():
    st.markdown('<h2 class="section-header">⚙️ Detection Settings</h2>', unsafe_allow_html=True)
    
    if not st.session_state.model_loaded:
        st.error("❌ Please select classes first on the Class Selection page.")
        return
    
    if not st.session_state.selected_ppe:
        st.warning("⚠️ No classes selected. Please go to 'Class Selection' page first.")
        return
    
    st.subheader("Detection Performance Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=st.session_state.detection_settings.get('confidence', 0.5),
            step=0.1,
            help="Higher values = fewer but more accurate detections"
        )
    
    with col2:
        speed_setting = st.selectbox(
            "Detection Speed",
            options=["fast", "medium", "accurate"],
            index=1,
            help="Fast: Lower accuracy, Medium: Balanced, Accurate: Higher accuracy but slower"
        )
    
    with col3:
        frame_skip = st.slider(
            "Frame Processing Rate",
            min_value=1,
            max_value=10,
            value=st.session_state.detection_settings.get('frame_skip', 3),
            help="Process every Nth frame (1=process all frames, 10=process every 10th frame)"
        )
    
    # Map speed settings
    speed_params = {
        "fast": {"imgsz": 320, "half": False},
        "medium": {"imgsz": 640, "half": False},
        "accurate": {"imgsz": 1280, "half": False}
    }
    
    # Save settings
    if st.button("💾 Save Settings", type="primary"):
        st.session_state.detection_settings = {
            'confidence': confidence,
            'speed': speed_setting,
            'frame_skip': frame_skip,
            'speed_params': speed_params[speed_setting]
        }
        st.success("✅ Settings saved successfully!")
    
    # Show current selection
    st.subheader("Currently Selected Classes")
    selected_classes_display = [f"{class_id}: {class_name}" for class_id, class_name in st.session_state.selected_ppe.items()]
    st.info(f"**Monitoring {len(selected_classes_display)} classes:** {', '.join(selected_classes_display[:5])}{'...' if len(selected_classes_display) > 5 else ''}")

def show_live_monitoring():
    st.markdown('<h2 class="section-header">📹 Live Safety Monitoring</h2>', unsafe_allow_html=True)
    
    if not st.session_state.model_loaded:
        st.error("❌ Please wait for model initialization on the Class Selection page first.")
        return
    
    if not st.session_state.selected_ppe:
        st.warning("⚠️ Please select classes to monitor first on the Class Selection page!")
        return
    
    # Display current configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Monitoring Controls")
        
        # Camera options - remove local webcam for cloud compatibility
        camera_mode = st.radio(
            "Select Input Source:",
            ["IP Camera", "Test Mode", "Upload Video"],
            help="Choose your video source"
        )
        
        # IP Camera selection
        if camera_mode == "IP Camera":
            if st.session_state.camera_urls:
                selected_url = st.selectbox(
                    "Select Saved Camera:",
                    st.session_state.camera_urls,
                    help="Choose from your saved camera URLs"
                )
                new_url = st.text_input("Or enter new camera URL:")
                camera_url = new_url if new_url else selected_url
            else:
                camera_url = st.text_input(
                    "Enter IP Camera URL:",
                    placeholder="rtsp://username:password@ip:port/stream"
                )
        
        # Performance info
        st.info(f"""
        **Current Settings:**
        - Monitoring: {len(st.session_state.selected_ppe)} classes
        - Confidence: {st.session_state.detection_settings.get('confidence', 0.5)}
        - Speed: {st.session_state.detection_settings.get('speed', 'medium').title()}
        - Frame Skip: {st.session_state.detection_settings.get('frame_skip', 3)}
        """)
        
        # Start buttons
        if camera_mode == "IP Camera":
            if camera_url:
                if st.button("🌐 Start IP Camera", type="primary"):
                    if CV2_AVAILABLE:
                        st.session_state.monitoring = True
                        start_ip_camera_monitoring(camera_url)
                    else:
                        st.error("❌ OpenCV not available for camera streaming")
                
                if st.button("⏹️ Stop Monitoring"):
                    st.session_state.monitoring = False
                    st.rerun()
            else:
                st.warning("Please enter an IP camera URL first")
                
        elif camera_mode == "Test Mode":
            if st.button("🧪 Start Test Mode", type="primary"):
                st.session_state.monitoring = True
                start_test_mode()
            
            if st.button("⏹️ Stop Test Mode"):
                st.session_state.monitoring = False
                st.rerun()
                
        elif camera_mode == "Upload Video":
            uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
            if uploaded_file and st.button("🎬 Process Video"):
                if CV2_AVAILABLE:
                    process_uploaded_video(uploaded_file)
                else:
                    st.error("❌ OpenCV not available for video processing")
    
    with col2:
        st.subheader("Live Stats")
        st.metric("Violations Detected", len(st.session_state.violations))
        st.metric("Monitoring Status", "ACTIVE" if st.session_state.monitoring else "INACTIVE")
        st.metric("Selected Classes", len(st.session_state.selected_ppe))
        st.metric("Model Mode", "DEMO" if st.session_state.demo_mode else "PRODUCTION")

def start_ip_camera_monitoring(camera_url):
    """IP camera monitoring"""
    if not CV2_AVAILABLE:
        st.error("❌ OpenCV not available for camera streaming")
        return
        
    st.info(f"🌐 Connecting to IP camera: {camera_url}")
    
    confidence = st.session_state.detection_settings.get('confidence', 0.5)
    frame_skip = st.session_state.detection_settings.get('frame_skip', 3)
    speed_params = st.session_state.detection_settings.get('speed_params', {'imgsz': 640, 'half': False})
    selected_classes = list(st.session_state.selected_ppe.keys())
    
    # Test connection first
    success, message = test_ip_camera(camera_url)
    if not success:
        st.error(f"❌ {message}")
        st.session_state.monitoring = False
        return
    
    st.success(message)
    
    # Open camera stream
    cap = cv2.VideoCapture(camera_url)
    
    if not cap.isOpened():
        st.error("❌ Failed to open camera stream")
        st.session_state.monitoring = False
        return
    
    run_monitoring_loop(cap, f"IP Camera: {camera_url}", selected_classes, confidence, frame_skip, speed_params)

def run_monitoring_loop(cap, source_name, selected_classes, confidence, frame_skip, speed_params):
    """Generic monitoring loop for all camera types"""
    if not CV2_AVAILABLE:
        st.error("❌ OpenCV not available for monitoring")
        return
        
    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    performance_placeholder = st.empty()
    
    frame_count = 0
    processing_times = []
    last_fps_update = time.time()
    fps = 0
    
    while st.session_state.monitoring and cap.isOpened():
        try:
            start_time = time.time()
            ret, frame = cap.read()
            
            if not ret:
                st.error("❌ Failed to read frame from camera")
                break
            
            frame_count += 1
            
            if frame_count % frame_skip == 0:
                # Run detection
                results = st.session_state.model(
                    frame, 
                    conf=confidence,
                    classes=selected_classes,
                    verbose=False,
                    **speed_params
                )
                
                violations = check_for_violations(results, selected_classes)
                annotated_frame = results[0].plot()
                
                # Add performance overlay
                cv2.putText(annotated_frame, f"SafetyEagle AI", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(annotated_frame, f"Classes: {len(selected_classes)}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(annotated_frame_rgb, caption=source_name, use_column_width=True)
                
                if violations:
                    status_placeholder.warning(f"🚨 Missing: {', '.join(violations)}")
                    save_violation(frame, violations)
                else:
                    status_placeholder.success("✅ All selected classes detected")
            
            # Calculate FPS
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            if time.time() - last_fps_update > 1.0:
                if processing_times:
                    avg_time = np.mean(processing_times)
                    fps = 1.0 / avg_time if avg_time > 0 else 0
                    processing_times = []
                last_fps_update = time.time()
                
                performance_placeholder.info(
                    f"**Performance:** {fps:.1f} FPS | "
                    f"Frame skip: {frame_skip} | "
                    f"Processing: {avg_time*1000:.1f}ms"
                )
            
            time.sleep(0.01)
            
        except Exception as e:
            st.error(f"Monitoring error: {e}")
            break
    
    cap.release()

def start_test_mode():
    """Test mode with simulation"""
    st.success("🎯 SafetyEagle Test Mode Active - Class Detection Simulation")
    
    selected_classes = list(st.session_state.selected_ppe.keys())
    
    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    info_placeholder = st.empty()
    
    frame_count = 0
    
    while st.session_state.monitoring:
        try:
            test_image = create_custom_test_image(frame_count, st.session_state.selected_ppe)
            
            if st.session_state.model:
                results = st.session_state.model(
                    test_image, 
                    conf=st.session_state.detection_settings.get('confidence', 0.5),
                    classes=selected_classes,
                    verbose=False
                )
                annotated_frame = results[0].plot()
            else:
                # Fallback if no model
                annotated_frame = test_image
            
            cv2.putText(annotated_frame, "SAFETYEAGLE TEST MODE", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(annotated_frame_rgb, caption="SafetyEagle Test Mode", use_column_width=True)
            
            # Simulate violations for demo
            if frame_count % 50 < 25:
                status_placeholder.warning("🚨 Simulation: Safety violation detected")
                info_placeholder.info("Demo: Missing safety equipment simulated")
                if frame_count % 30 == 0:
                    save_violation(test_image, ["Simulated Violation"])
            else:
                status_placeholder.success("✅ Simulation: All safety protocols followed")
                info_placeholder.info("Demo: Normal operation simulated")
            
            frame_count += 1
            time.sleep(0.3)
            
        except Exception as e:
            st.error(f"Test mode error: {e}")
            break

def create_custom_test_image(frame_count, selected_ppe):
    """Create test image based on selected classes"""
    img = np.ones((480, 640, 3), dtype=np.uint8) * 150
    cv2.rectangle(img, (200, 100), (440, 400), (0, 255, 0), 2)
    cv2.putText(img, "SafetyEagle AI", (250, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    class_names = list(selected_ppe.values())
    scenario = (frame_count // 40) % (len(class_names) + 1)
    missing_items = []
    
    if scenario > 0:
        missing_index = (scenario - 1) % len(class_names)
        missing_items = [class_names[missing_index]]
    
    y_pos = 50
    for i, class_name in enumerate(class_names):
        if class_name not in missing_items:
            color = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)][i % 5]
            cv2.rectangle(img, (250, y_pos), (390, y_pos + 40), color, -1)
            cv2.putText(img, class_name, (260, y_pos + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 50
    
    if missing_items:
        cv2.putText(img, f"MISSING: {', '.join(missing_items)}", 
                   (200, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return img

def process_uploaded_video(uploaded_file):
    """Process uploaded video"""
    if not CV2_AVAILABLE:
        st.error("❌ OpenCV not available for video processing")
        return
        
    confidence = st.session_state.detection_settings.get('confidence', 0.5)
    selected_classes = list(st.session_state.selected_ppe.keys())
    speed_params = st.session_state.detection_settings.get('speed_params', {'imgsz': 640, 'half': False})
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name
    
    cap = cv2.VideoCapture(video_path)
    frame_placeholder = st.empty()
    progress_bar = st.progress(0)
    status_placeholder = st.empty()
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if st.session_state.model:
            results = st.session_state.model(
                frame, 
                conf=confidence,
                classes=selected_classes,
                verbose=False,
                **speed_params
            )
            
            violations = check_for_violations(results, selected_classes)
            annotated_frame = results[0].plot()
        else:
            # Fallback if no model
            annotated_frame = frame
            violations = []
        
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        frame_placeholder.image(annotated_frame_rgb, caption="SafetyEagle Video Analysis", use_column_width=True)
        
        if violations:
            status_placeholder.warning(f"Violations: {', '.join(violations)}")
            save_violation(frame, violations)
        else:
            status_placeholder.info("No violations detected")
        
        frame_count += 1
        progress_bar.progress(frame_count / total_frames)
        time.sleep(0.03)
    
    cap.release()
    os.unlink(video_path)
    st.success("✅ Video analysis completed!")

def check_for_violations(results, required_classes):
    """Check for class detection violations"""
    detected_classes = set()
    
    if results and len(results) > 0:
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            detected_classes.add(class_id)
    
    missing_classes = []
    for class_id in required_classes:
        if class_id not in detected_classes:
            class_name = st.session_state.selected_ppe.get(class_id, f"Class {class_id}")
            missing_classes.append(class_name)
    
    return missing_classes

def save_violation(frame, violations):
    """Save violation record"""
    violation_record = {
        'timestamp': datetime.now(),
        'missing_classes': ', '.join(violations),
        'image': frame.copy(),
        'selected_classes': list(st.session_state.selected_ppe.values())
    }
    st.session_state.violations.append(violation_record)

def show_dashboard():
    st.markdown('<h2 class="section-header">📊 Safety Dashboard</h2>', unsafe_allow_html=True)
    
    if not st.session_state.violations:
        st.info("No safety violations recorded yet. Start monitoring to see data.")
        return
    
    df = pd.DataFrame([
        {
            'timestamp': v['timestamp'],
            'missing_classes': v['missing_classes'],
            'hour': v['timestamp'].hour,
            'selected_classes': ', '.join(v['selected_classes'])
        }
        for v in st.session_state.violations
    ])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Violations", len(st.session_state.violations))
    
    with col2:
        st.metric("Today's Violations", len(df))
    
    with col3:
        most_common = df['missing_classes'].mode()[0] if not df.empty else "None"
        st.metric("Most Common Issue", most_common)
    
    with col4:
        current_hour = datetime.now().hour
        hour_violations = len(df[df['hour'] == current_hour])
        st.metric("This Hour", hour_violations)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Violations by Hour")
        if not df.empty:
            hourly_data = df.groupby('hour').size()
            fig = px.bar(
                x=hourly_data.index,
                y=hourly_data.values,
                labels={'x': 'Hour of Day', 'y': 'Violations'},
                title="Safety Violations by Hour"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Violation Distribution")
        class_counts = {}
        for missing in df['missing_classes']:
            items = missing.split(', ')
            for item in items:
                class_counts[item] = class_counts.get(item, 0) + 1
        
        if class_counts:
            fig = px.pie(
                values=list(class_counts.values()),
                names=list(class_counts.keys()),
                title="Safety Violation Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Recent Safety Violations")
    for i, violation in enumerate(st.session_state.violations[-5:]):
        with st.expander(f"Violation {i+1} - {violation['timestamp'].strftime('%H:%M:%S')}"):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(violation['image'], use_column_width=True)
            with col2:
                st.write(f"**Missing Safety Items:** {violation['missing_classes']}")
                st.write(f"**Time:** {violation['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")

def show_reports():
    st.markdown('<h2 class="section-header">📈 Safety Reports & Analytics</h2>', unsafe_allow_html=True)
    
    if not st.session_state.violations:
        st.warning("No safety data available for reports. Start monitoring first.")
        return
    
    st.info(f"**Current Safety Monitoring:** {len(st.session_state.selected_ppe)} classes selected")
    
    if st.button("📊 Generate Excel Report", type="primary"):
        generate_excel_report()
    
    df = pd.DataFrame([
        {
            'Timestamp': v['timestamp'],
            'Missing Safety Items': v['missing_classes'],
            'Monitored Classes': v['selected_classes'],
            'Date': v['timestamp'].date(),
            'Time': v['timestamp'].time()
        }
        for v in st.session_state.violations
    ])
    
    st.dataframe(df, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV Report",
            csv,
            "safetyeagle_violations.csv",
            "text/csv"
        )
    
    with col2:
        if st.button("🗑️ Clear All Data"):
            st.session_state.violations = []
            st.rerun()

def generate_excel_report():
    """Generate Excel report"""
    df = pd.DataFrame([
        {
            'Timestamp': v['timestamp'],
            'Missing_Safety_Items': v['missing_classes'],
            'Monitored_Classes': v['selected_classes'],
            'Date': v['timestamp'].date(),
            'Time': v['timestamp'].time(),
            'Hour': v['timestamp'].hour
        }
        for v in st.session_state.violations
    ])
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Safety_Violations', index=False)
        
        summary_data = {
            'Report_Generated': [datetime.now()],
            'Total_Violations': [len(df)],
            'Monitoring_Configuration': [f"{len(st.session_state.selected_ppe)} safety classes"],
            'Most_Common_Violation': [df['Missing_Safety_Items'].mode()[0] if not df.empty else 'None'],
            'Confidence_Setting': [st.session_state.detection_settings.get('confidence', 0.5)],
            'Speed_Setting': [st.session_state.detection_settings.get('speed', 'medium')]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Executive_Summary', index=False)
        
        config_data = {
            'Class_ID': list(st.session_state.selected_ppe.keys()),
            'Class_Name': list(st.session_state.selected_ppe.values())
        }
        pd.DataFrame(config_data).to_excel(writer, sheet_name='Safety_Configuration', index=False)
    
    st.download_button(
        "📥 Download SafetyEagle Report",
        output.getvalue(),
        f"safetyeagle_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

def show_deployment_guide():
    st.markdown('<h2 class="section-header">🌐 SafetyEagle Deployment Guide</h2>', unsafe_allow_html=True)
    
    st.info("🦅 **SafetyEagle AI** - Successfully deployed on Streamlit Cloud!")
    
    st.subheader("🚀 Deployment Status: ✅ LIVE")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **✅ Current Features:**
        - Class selection from YOLO model
        - IP camera integration
        - Video file analysis
        - Safety violation tracking
        - Professional reporting
        - Demo/test mode
        """)
    
    with col2:
        st.markdown("""
        **🔧 Technical Stack:**
        - Streamlit Cloud hosting
        - YOLOv8 object detection
        - OpenCV for video processing
        - Plotly for analytics
        - Pandas for data handling
        """)
    
    with col3:
        st.markdown("""
        **📊 Available Now:**
        - Real-time monitoring
        - Safety dashboards
        - Compliance reports
        - Multi-class detection
        - Professional UI/UX
        """)
    
    st.subheader("🎯 Next Steps for Enhanced Deployment")
    
    st.markdown("""
    **For Full Production Deployment:**
    
    1. **Custom Model Upload:**
       ```python
       # Upload your trained model to:
       models/best.pt
       ```
    
    2. **Enhanced Requirements:**
       ```txt
       # In requirements.txt
       opencv-python-headless==4.8.1.78
       ultralytics==8.0.186
       torch==2.0.1
       ```
    
    3. **Streamlit Configuration:**
       ```toml
       # In .streamlit/config.toml
       [server]
       headless = true
       address = "0.0.0.0"
       port = 8501
       ```
    """)
    
    st.success("🎉 **SafetyEagle AI is now successfully deployed and operational!**")

if __name__ == "__main__":
    main()


