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
import io
import requests
import re

# Set page configuration
st.set_page_config(
    page_title="PPE Safety Monitor",
    page_icon="🛡️",
    layout="wide"
)

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

# Load model with caching and get all classes
@st.cache_resource
def load_model_and_classes():
    try:
        # Try multiple possible model paths
        possible_paths = [
            "models/best.pt",
            "best.pt", 
            "./models/best.pt",
            r"D:\runs\detect\train\weights\best.pt"
        ]
        
        model = None
        loaded_path = None
        
        for model_path in possible_paths:
            if os.path.exists(model_path):
                model = YOLO(model_path)
                loaded_path = model_path
                st.success(f"✅ Model loaded from: {model_path}")
                break
        
        if model is None:
            # Fallback to nano model
            st.warning("⚠️ Local model not found. Using YOLOv8n for demo...")
            model = YOLO('yolov8n.pt')
            loaded_path = 'yolov8n.pt'
        
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
        return None, {}, None

def initialize_app():
    """Initialize the app and load model on startup"""
    if not st.session_state.model_loaded:
        with st.spinner("🔄 Loading AI model and detecting available classes..."):
            model, available_classes, model_path = load_model_and_classes()
            st.session_state.model = model
            st.session_state.available_classes = available_classes
            st.session_state.model_loaded = True
            
            # Auto-select all classes by default
            if available_classes:
                st.session_state.selected_ppe = available_classes.copy()
                st.success(f"✅ Loaded {len(available_classes)} classes from model")

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
    # Initialize app on startup
    initialize_app()
    
    st.title("🌐 PPE Safety Monitoring System")
    
    # Show model status in sidebar
    st.sidebar.title("Model Status")
    if st.session_state.model_loaded and st.session_state.available_classes:
        st.sidebar.success(f"✅ Model Loaded")
        st.sidebar.info(f"**Available Classes:** {len(st.session_state.available_classes)}")
        
        # Show quick class overview
        with st.sidebar.expander("📋 Quick Class Overview"):
            for class_id, class_name in list(st.session_state.available_classes.items())[:10]:  # Show first 10
                status = "✅" if class_id in st.session_state.selected_ppe else "❌"
                st.write(f"{status} **{class_id}:** {class_name}")
            if len(st.session_state.available_classes) > 10:
                st.write(f"... and {len(st.session_state.available_classes) - 10} more classes")
    else:
        st.sidebar.error("❌ Model Not Loaded")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Class Selection", "Camera Setup", "Settings", "Live Monitoring", "Dashboard", "Reports", "Deployment"])
    
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
    elif page == "Deployment":
        show_deployment_guide()

def show_class_selection():
    st.header("🎯 Select Classes to Detect")
    
    if not st.session_state.model_loaded:
        st.error("❌ Model not loaded. Please wait for model initialization or check the model path.")
        if st.button("🔄 Retry Model Loading"):
            st.session_state.model_loaded = False
            st.rerun()
        return
    
    if not st.session_state.available_classes:
        st.error("❌ No classes available from the model. Please check your model file.")
        return
    
    st.success(f"🎉 **Model loaded successfully! Found {len(st.session_state.available_classes)} classes**")
    
    # Display all available classes for selection
    st.subheader("Available Detection Classes")
    st.info("Select which classes you want to monitor for PPE compliance:")
    
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
    
    # Model information
    with st.expander("🔧 Model Information"):
        st.write(f"**Total Classes Available:** {len(st.session_state.available_classes)}")
        st.write(f"**Currently Selected:** {len(selected_classes)}")
        st.write("**All Available Classes:**")
        for class_id, class_name in st.session_state.available_classes.items():
            st.write(f"- **{class_id}:** {class_name}")

def show_camera_setup():
    st.header("📷 Camera Configuration")
    
    if not st.session_state.model_loaded:
        st.error("❌ Please wait for model initialization on the Class Selection page first.")
        return
    
    st.subheader("1. Local Webcam")
    st.info("Use your computer's built-in or USB camera")
    
    if st.button("🔍 Detect Local Cameras"):
        detect_local_cameras()
    
    st.subheader("2. IP Camera / Network Stream")
    st.info("Connect to IP cameras, RTSP streams, or network cameras")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        camera_url = st.text_input(
            "IP Camera URL:",
            placeholder="rtsp://username:password@ip:port/stream or http://ip:port/video"
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        if st.button("🔗 Test Connection"):
            if camera_url:
                is_valid, message = validate_ip_camera_url(camera_url)
                if is_valid:
                    with st.spinner("Testing camera connection..."):
                        success, result = test_ip_camera(camera_url)
                    if success:
                        st.success(result)
                        # Add to saved URLs if not already there
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
        - `rtsp://192.168.1.100:8554/live`
        
        **HTTP Examples:**
        - `http://192.168.1.100:8080/video`
        - `http://192.168.1.100:4747/video`
        - `http://ip_address:port/stream`
        
        **Popular Camera Brands:**
        - **Hikvision:** `rtsp://admin:password@ip:554/Streaming/Channels/101`
        - **Dahua:** `rtsp://admin:password@ip:554/cam/realmonitor?channel=1&subtype=0`
        - **Axis:** `rtsp://root:pass@ip/axis-media/media.amp`
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
    
    st.subheader("3. Mobile Phone as Camera")
    st.info("Use your smartphone as a wireless camera")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **For Android:**
        1. Install **IP Webcam** app
        2. Start server in the app
        3. Note the IP address shown
        4. Use URL like: `http://192.168.1.100:8080/video`
        """)
    
    with col2:
        st.markdown("""
        **For iPhone:**
        1. Install **IP Camera** app
        2. Configure the stream settings
        3. Start the server
        4. Use the provided RTSP or HTTP URL
        """)
    
    st.subheader("4. Video File Upload")
    st.info("Use pre-recorded videos for testing")
    
    uploaded_file = st.file_uploader("Upload video file", type=['mp4', 'avi', 'mov', 'mkv'])
    if uploaded_file:
        st.success(f"✅ Video uploaded: {uploaded_file.name}")

def detect_local_cameras():
    """Detect available local cameras"""
    st.info("Scanning for local cameras...")
    
    available_cameras = []
    max_cameras_to_check = 5
    
    for i in range(max_cameras_to_check):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                available_cameras.append(i)
                st.success(f"📷 Camera found at index {i} - {frame.shape[1]}x{frame.shape[0]}")
                
                # Show preview
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, caption=f"Camera {i} Preview", use_column_width=True)
            cap.release()
        else:
            st.warning(f"❌ No camera at index {i}")
    
    if available_cameras:
        st.success(f"🎯 Found {len(available_cameras)} camera(s): {available_cameras}")
    else:
        st.error("❌ No local cameras found!")

def show_settings():
    st.header("⚙️ Detection Settings")
    
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
        "fast": {"imgsz": 320, "half": True},
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
        
        # Show summary
        st.subheader("Current Configuration:")
        st.write(f"**Selected Classes:** {len(st.session_state.selected_ppe)}")
        st.write(f"**Confidence:** {confidence}")
        st.write(f"**Speed:** {speed_setting.title()}")
        st.write(f"**Frame Skip:** {frame_skip}")
    
    # Show current selection
    st.subheader("Currently Selected Classes")
    selected_classes_display = [f"{class_id}: {class_name}" for class_id, class_name in st.session_state.selected_ppe.items()]
    st.info(f"**Monitoring {len(selected_classes_display)} classes:** {', '.join(selected_classes_display[:5])}{'...' if len(selected_classes_display) > 5 else ''}")

def show_live_monitoring():
    st.header("📹 Live Monitoring")
    
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
        
        # Camera options
        camera_mode = st.radio(
            "Select Input Source:",
            ["Local Webcam", "IP Camera", "Test Mode", "Upload Video"],
            help="Choose your camera source"
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
        if camera_mode == "Local Webcam":
            if st.button("🎥 Start Webcam", type="primary"):
                st.session_state.monitoring = True
                start_webcam_monitoring()
            
            if st.button("⏹️ Stop Monitoring"):
                st.session_state.monitoring = False
                st.rerun()
                
        elif camera_mode == "IP Camera":
            if camera_url:
                if st.button("🌐 Start IP Camera", type="primary"):
                    st.session_state.monitoring = True
                    start_ip_camera_monitoring(camera_url)
                
                if st.button("⏹️ Stop Monitoring"):
                    st.session_state.monitoring = False
                    st.rerun()
            else:
                st.warning("Please enter an IP camera URL first")
                
        elif camera_mode == "Test Mode":
            if st.button("🔄 Start Test Mode", type="primary"):
                st.session_state.monitoring = True
                start_test_mode()
            
            if st.button("⏹️ Stop Test Mode"):
                st.session_state.monitoring = False
                st.rerun()
                
        elif camera_mode == "Upload Video":
            uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
            if uploaded_file and st.button("🎬 Process Video"):
                process_uploaded_video(uploaded_file)
    
    with col2:
        st.subheader("Live Stats")
        st.metric("Violations Detected", len(st.session_state.violations))
        st.metric("Monitoring Status", "ACTIVE" if st.session_state.monitoring else "INACTIVE")
        st.metric("Selected Classes", len(st.session_state.selected_ppe))
        st.metric("Total Available Classes", len(st.session_state.available_classes))

def start_webcam_monitoring():
    """Local webcam monitoring"""
    st.info("🚀 Starting local webcam monitoring...")
    
    confidence = st.session_state.detection_settings.get('confidence', 0.5)
    frame_skip = st.session_state.detection_settings.get('frame_skip', 3)
    speed_params = st.session_state.detection_settings.get('speed_params', {'imgsz': 640, 'half': False})
    selected_classes = list(st.session_state.selected_ppe.keys())
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        st.error("❌ Cannot access webcam. Switching to Test Mode...")
        start_test_mode()
        return
    
    # Optimize camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    run_monitoring_loop(cap, "Local Webcam", selected_classes, confidence, frame_skip, speed_params)

def start_ip_camera_monitoring(camera_url):
    """IP camera monitoring"""
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
    if camera_url.startswith('rtsp://'):
        cap = cv2.VideoCapture(camera_url)
    else:
        cap = cv2.VideoCapture(camera_url)
    
    if not cap.isOpened():
        st.error("❌ Failed to open camera stream")
        st.session_state.monitoring = False
        return
    
    run_monitoring_loop(cap, f"IP Camera: {camera_url}", selected_classes, confidence, frame_skip, speed_params)

def run_monitoring_loop(cap, source_name, selected_classes, confidence, frame_skip, speed_params):
    """Generic monitoring loop for all camera types"""
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
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Source: {source_name}", (10, 60), 
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
    st.success("🎯 Test Mode Active - Class Detection Simulation")
    
    selected_classes = list(st.session_state.selected_ppe.keys())
    
    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    info_placeholder = st.empty()
    
    frame_count = 0
    
    while st.session_state.monitoring:
        try:
            test_image = create_custom_test_image(frame_count, st.session_state.selected_ppe)
            results = st.session_state.model(
                test_image, 
                conf=st.session_state.detection_settings.get('confidence', 0.5),
                classes=selected_classes,
                verbose=False
            )
            
            violations = check_for_violations(results, selected_classes)
            annotated_frame = results[0].plot()
            
            cv2.putText(annotated_frame, "TEST MODE", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(annotated_frame_rgb, caption="Test Mode - Class Detection", use_column_width=True)
            
            if violations:
                status_placeholder.warning(f"🚨 Missing: {', '.join(violations)}")
                info_placeholder.info("Simulation: Some classes not detected")
                if frame_count % 30 == 0:
                    save_violation(test_image, violations)
            else:
                status_placeholder.success("✅ All selected classes detected")
                info_placeholder.info("Simulation: All selected classes present")
            
            frame_count += 1
            time.sleep(0.3)
            
        except Exception as e:
            st.error(f"Test mode error: {e}")
            break

def create_custom_test_image(frame_count, selected_ppe):
    """Create test image based on selected classes"""
    img = np.ones((480, 640, 3), dtype=np.uint8) * 150
    cv2.rectangle(img, (200, 100), (440, 400), (0, 255, 0), 2)
    cv2.putText(img, "Detection Area", (250, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
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
        
        results = st.session_state.model(
            frame, 
            conf=confidence,
            classes=selected_classes,
            verbose=False,
            **speed_params
        )
        
        violations = check_for_violations(results, selected_classes)
        annotated_frame = results[0].plot()
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        frame_placeholder.image(annotated_frame_rgb, caption="Video Processing", use_column_width=True)
        
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
    st.success("Video processing completed!")

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
    st.header("📊 Safety Dashboard")
    
    if not st.session_state.violations:
        st.info("No violations recorded yet. Start monitoring to see data.")
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
        hourly_data = df.groupby('hour').size()
        fig = px.bar(
            x=hourly_data.index,
            y=hourly_data.values,
            labels={'x': 'Hour of Day', 'y': 'Violations'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Missing Classes Distribution")
        class_counts = {}
        for missing in df['missing_classes']:
            items = missing.split(', ')
            for item in items:
                class_counts[item] = class_counts.get(item, 0) + 1
        
        if class_counts:
            fig = px.pie(
                values=list(class_counts.values()),
                names=list(class_counts.keys()),
                title="Missing Class Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Recent Violations")
    for i, violation in enumerate(st.session_state.violations[-5:]):
        with st.expander(f"Violation {i+1} - {violation['timestamp'].strftime('%H:%M:%S')}"):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(violation['image'], use_column_width=True)
            with col2:
                st.write(f"**Missing:** {violation['missing_classes']}")
                st.write(f"**Monitoring:** {violation['selected_classes']}")
                st.write(f"**Time:** {violation['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")

def show_reports():
    st.header("📈 Reports & Analytics")
    
    if not st.session_state.violations:
        st.warning("No data available for reports. Start monitoring first.")
        return
    
    st.info(f"**Current Monitoring Configuration:** {len(st.session_state.selected_ppe)} classes selected")
    
    if st.button("📊 Generate Excel Report", type="primary"):
        generate_excel_report()
    
    df = pd.DataFrame([
        {
            'Timestamp': v['timestamp'],
            'Missing Classes': v['missing_classes'],
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
            "📥 Download CSV",
            csv,
            "class_violations.csv",
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
            'Missing_Classes': v['missing_classes'],
            'Monitored_Classes': v['selected_classes'],
            'Date': v['timestamp'].date(),
            'Time': v['timestamp'].time(),
            'Hour': v['timestamp'].hour
        }
        for v in st.session_state.violations
    ])
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Violations', index=False)
        
        summary_data = {
            'Total_Violations': [len(df)],
            'Date_Generated': [datetime.now()],
            'Monitoring_Configuration': [f"{len(st.session_state.selected_ppe)} classes"],
            'Most_Common_Violation': [df['Missing_Classes'].mode()[0] if not df.empty else 'None'],
            'Confidence_Setting': [st.session_state.detection_settings.get('confidence', 0.5)],
            'Speed_Setting': [st.session_state.detection_settings.get('speed', 'medium')]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        config_data = {
            'Class_ID': list(st.session_state.selected_ppe.keys()),
            'Class_Name': list(st.session_state.selected_ppe.values())
        }
        pd.DataFrame(config_data).to_excel(writer, sheet_name='Configuration', index=False)
    
    st.download_button(
        "📥 Download Excel Report",
        output.getvalue(),
        f"class_monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

def show_deployment_guide():
    st.header("🌐 Deployment Guide")
    
    st.subheader("1. Free Deployment Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Streamlit Community Cloud**
        - ✅ Completely free
        - ✅ Easy deployment
        - ✅ Public access
        - ❌ Limited resources
        - ❌ No GPU acceleration
        
        [Deploy Now](https://streamlit.io/cloud)
        """)
    
    with col2:
        st.markdown("""
        **Hugging Face Spaces**
        - ✅ Free tier available
        - ✅ Easy Git integration
        - ✅ Community sharing
        - ❌ Limited compute
        - ❌ Slower startup
        
        [Deploy Now](https://huggingface.co/spaces)
        """)
    
    with col3:
        st.markdown("""
        **Railway**
        - ✅ Free tier available
        - ✅ Easy deployment
        - ✅ Good performance
        - ❌ Limited hours/month
        - ❌ Credit card required
        
        [Deploy Now](https://railway.app)
        """)
    
    st.subheader("2. Step-by-Step Deployment (Streamlit Cloud)")
    
    st.markdown("""
    **Step 1: Prepare Your Files**
    ```
    your-project-folder/
    ├── ppe.py                    # This Streamlit app
    ├── requirements.txt          # Dependencies list
    ├── models/
    │   └── best.pt              # Your trained model
    ├── .streamlit/
    │   └── config.toml          # Streamlit config
    └── README.md                # Optional description
    ```
    
    **Step 2: Create requirements.txt**
    ```txt
    streamlit==1.28.0
    opencv-python-headless==4.8.1.78
    ultralytics==8.0.186
    pandas==2.1.0
    numpy==1.24.3
    Pillow==10.0.1
    plotly==5.15.0
    requests==2.31.0
    torch==2.0.1
    ```
    
    **Step 3: Deploy to Streamlit Cloud**
    1. Go to [share.streamlit.io](https://share.streamlit.io)
    2. Sign in with GitHub
    3. Connect your repository
    4. Set main file path to `ppe.py`
    5. Click "Deploy"
    """)
    
    st.info("🎯 **New Feature:** The app now automatically loads all available classes from your model and lets you select which ones to monitor!")

if __name__ == "__main__":
    main()