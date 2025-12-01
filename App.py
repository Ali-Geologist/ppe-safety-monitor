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

# Load model with caching
@st.cache_resource
def load_model():
    try:
        model = YOLO(r"D:\runs\detect\train\weights\best.pt")
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None

def get_available_classes(model):
    """Get available class names from the model"""
    if model and hasattr(model, 'names'):
        return model.names
    return {}

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
    st.title("🌐 PPE Safety Monitoring System")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Camera Setup", "Settings", "Live Monitoring", "Dashboard", "Reports", "Deployment"])
    
    if page == "Camera Setup":
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

def show_camera_setup():
    st.header("📷 Camera Configuration")
    
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
    
    # Initialize model first
    if st.session_state.model is None:
        st.session_state.model = load_model()
    
    if st.session_state.model is None:
        st.error("Could not load model. Please check the model path.")
        return
    
    # Get available classes
    available_classes = get_available_classes(st.session_state.model)
    
    if not available_classes:
        st.error("No classes found in the model. Please check your model.")
        return
    
    st.subheader("1. Select PPE to Detect")
    st.info("Choose which safety equipment you want to monitor:")
    
    # PPE selection
    col1, col2, col3 = st.columns(3)
    
    # Common PPE items mapping
    ppe_mapping = {
        'Hard Hat/Helmet': 'helmet',
        'Safety Vest': 'vest', 
        'Safety Gloves': 'gloves',
        'Safety Glasses': 'glasses',
        'Safety Boots': 'boots',
        'Face Mask': 'mask',
        'Person': 'person'
    }
    
    selected_ppe = {}
    
    with col1:
        for ppe_display in list(ppe_mapping.keys())[:3]:
            ppe_key = ppe_mapping[ppe_display]
            matching_classes = [cls_id for cls_id, name in available_classes.items() 
                              if ppe_key in name.lower() or name.lower() in ppe_key]
            
            if matching_classes:
                class_id = matching_classes[0]
                is_selected = st.checkbox(
                    f"{ppe_display} (Class {class_id})", 
                    value=True,
                    key=f"ppe_{ppe_key}"
                )
                if is_selected:
                    selected_ppe[class_id] = ppe_display
    
    with col2:
        for ppe_display in list(ppe_mapping.keys())[3:5]:
            ppe_key = ppe_mapping[ppe_display]
            matching_classes = [cls_id for cls_id, name in available_classes.items() 
                              if ppe_key in name.lower() or name.lower() in ppe_key]
            
            if matching_classes:
                class_id = matching_classes[0]
                is_selected = st.checkbox(
                    f"{ppe_display} (Class {class_id})", 
                    value=True,
                    key=f"ppe_{ppe_key}"
                )
                if is_selected:
                    selected_ppe[class_id] = ppe_display
    
    with col3:
        for ppe_display in list(ppe_mapping.keys())[5:]:
            ppe_key = ppe_mapping[ppe_display]
            matching_classes = [cls_id for cls_id, name in available_classes.items() 
                              if ppe_key in name.lower() or name.lower() in ppe_key]
            
            if matching_classes:
                class_id = matching_classes[0]
                is_selected = st.checkbox(
                    f"{ppe_display} (Class {class_id})", 
                    value=True,
                    key=f"ppe_{ppe_key}"
                )
                if is_selected:
                    selected_ppe[class_id] = ppe_display
    
    # Show available classes
    with st.expander("📋 All Available Classes in Model"):
        st.write("Your model can detect these classes:")
        for class_id, class_name in available_classes.items():
            st.write(f"**Class {class_id}:** {class_name}")
    
    st.subheader("2. Detection Performance Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
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
            value=3,
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
        st.session_state.selected_ppe = selected_ppe
        st.session_state.detection_settings = {
            'confidence': confidence,
            'speed': speed_setting,
            'frame_skip': frame_skip,
            'speed_params': speed_params[speed_setting]
        }
        st.success("✅ Settings saved successfully!")
        
        # Show summary
        st.subheader("Current Configuration:")
        st.write(f"**Selected PPE:** {', '.join(selected_ppe.values())}")
        st.write(f"**Confidence:** {confidence}")
        st.write(f"**Speed:** {speed_setting.title()}")
        st.write(f"**Frame Skip:** {frame_skip}")
    
    if not selected_ppe:
        st.warning("⚠️ Please select at least one PPE item to monitor.")

def show_live_monitoring():
    st.header("📹 Live PPE Monitoring")
    
    if not st.session_state.selected_ppe:
        st.warning("⚠️ Please configure detection settings first!")
        st.info("Go to the **Settings** page to select which PPE items to monitor.")
        return
    
    if st.session_state.model is None:
        st.session_state.model = load_model()
    
    if st.session_state.model is None:
        st.error("Could not load model. Please check the model path.")
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
        - Monitoring: {', '.join(st.session_state.selected_ppe.values())}
        - Confidence: {st.session_state.detection_settings['confidence']}
        - Speed: {st.session_state.detection_settings['speed'].title()}
        - Frame Skip: {st.session_state.detection_settings['frame_skip']}
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
        st.metric("Selected PPE Items", len(st.session_state.selected_ppe))

def start_webcam_monitoring():
    """Local webcam monitoring"""
    st.info("🚀 Starting local webcam monitoring...")
    
    confidence = st.session_state.detection_settings['confidence']
    frame_skip = st.session_state.detection_settings['frame_skip']
    speed_params = st.session_state.detection_settings['speed_params']
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
    
    confidence = st.session_state.detection_settings['confidence']
    frame_skip = st.session_state.detection_settings['frame_skip']
    speed_params = st.session_state.detection_settings['speed_params']
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
                
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(annotated_frame_rgb, caption=source_name, use_column_width=True)
                
                if violations:
                    status_placeholder.warning(f"🚨 Missing: {', '.join(violations)}")
                    save_violation(frame, violations)
                else:
                    status_placeholder.success("✅ All required PPE detected")
            
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
    st.success("🎯 Test Mode Active - Custom PPE Detection Simulation")
    
    selected_ppe = st.session_state.selected_ppe
    selected_classes = list(selected_ppe.keys())
    
    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    info_placeholder = st.empty()
    
    frame_count = 0
    
    while st.session_state.monitoring:
        try:
            test_image = create_custom_test_image(frame_count, selected_ppe)
            results = st.session_state.model(
                test_image, 
                conf=st.session_state.detection_settings['confidence'],
                classes=selected_classes,
                verbose=False
            )
            
            violations = check_for_violations(results, selected_classes)
            annotated_frame = results[0].plot()
            
            cv2.putText(annotated_frame, "TEST MODE", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(annotated_frame_rgb, caption="Test Mode - Custom PPE Detection", use_column_width=True)
            
            if violations:
                status_placeholder.warning(f"🚨 Missing: {', '.join(violations)}")
                info_placeholder.info("Simulation: PPE violations detected")
                if frame_count % 30 == 0:
                    save_violation(test_image, violations)
            else:
                status_placeholder.success("✅ All required PPE detected")
                info_placeholder.info("Simulation: All safety equipment present")
            
            frame_count += 1
            time.sleep(0.3)
            
        except Exception as e:
            st.error(f"Test mode error: {e}")
            break

def create_custom_test_image(frame_count, selected_ppe):
    """Create test image based on selected PPE"""
    img = np.ones((480, 640, 3), dtype=np.uint8) * 150
    cv2.rectangle(img, (200, 100), (440, 400), (0, 255, 0), 2)
    cv2.putText(img, "Person", (250, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    ppe_names = list(selected_ppe.values())
    scenario = (frame_count // 40) % (len(ppe_names) + 1)
    missing_items = []
    
    if scenario > 0:
        missing_index = (scenario - 1) % len(ppe_names)
        missing_items = [ppe_names[missing_index]]
    
    y_pos = 50
    for i, ppe_name in enumerate(ppe_names):
        if ppe_name not in missing_items:
            color = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)][i % 5]
            cv2.rectangle(img, (250, y_pos), (390, y_pos + 40), color, -1)
            cv2.putText(img, ppe_name, (260, y_pos + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 50
    
    if missing_items:
        cv2.putText(img, f"MISSING: {', '.join(missing_items)}", 
                   (200, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return img

def process_uploaded_video(uploaded_file):
    """Process uploaded video"""
    confidence = st.session_state.detection_settings['confidence']
    selected_classes = list(st.session_state.selected_ppe.keys())
    speed_params = st.session_state.detection_settings['speed_params']
    
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
    """Check for PPE violations"""
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
    """Save violation record"""
    violation_record = {
        'timestamp': datetime.now(),
        'missing_ppe': ', '.join(violations),
        'image': frame.copy(),
        'selected_ppe': list(st.session_state.selected_ppe.values())
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
            'missing_ppe': v['missing_ppe'],
            'hour': v['timestamp'].hour,
            'selected_ppe': ', '.join(v['selected_ppe'])
        }
        for v in st.session_state.violations
    ])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Violations", len(st.session_state.violations))
    
    with col2:
        st.metric("Today's Violations", len(df))
    
    with col3:
        most_common = df['missing_ppe'].mode()[0] if not df.empty else "None"
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
        st.subheader("Missing PPE Distribution")
        ppe_counts = {}
        for missing in df['missing_ppe']:
            items = missing.split(', ')
            for item in items:
                ppe_counts[item] = ppe_counts.get(item, 0) + 1
        
        if ppe_counts:
            fig = px.pie(
                values=list(ppe_counts.values()),
                names=list(ppe_counts.keys()),
                title="Missing PPE Items"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Recent Violations")
    for i, violation in enumerate(st.session_state.violations[-5:]):
        with st.expander(f"Violation {i+1} - {violation['timestamp'].strftime('%H:%M:%S')}"):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(violation['image'], use_column_width=True)
            with col2:
                st.write(f"**Missing:** {violation['missing_ppe']}")
                st.write(f"**Monitoring:** {violation['selected_ppe']}")
                st.write(f"**Time:** {violation['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")

def show_reports():
    st.header("📈 Reports & Analytics")
    
    if not st.session_state.violations:
        st.warning("No data available for reports. Start monitoring first.")
        return
    
    st.info(f"**Current Monitoring Configuration:** {', '.join(st.session_state.selected_ppe.values())}")
    
    if st.button("📊 Generate Excel Report", type="primary"):
        generate_excel_report()
    
    df = pd.DataFrame([
        {
            'Timestamp': v['timestamp'],
            'Missing PPE': v['missing_ppe'],
            'Monitored PPE': v['selected_ppe'],
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
            "ppe_violations.csv",
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
            'Missing_PPE': v['missing_ppe'],
            'Monitored_PPE': v['selected_ppe'],
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
            'Monitoring_Configuration': [', '.join(st.session_state.selected_ppe.values())],
            'Most_Common_Violation': [df['Missing_PPE'].mode()[0] if not df.empty else 'None'],
            'Confidence_Setting': [st.session_state.detection_settings['confidence']],
            'Speed_Setting': [st.session_state.detection_settings['speed']]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        config_data = {
            'PPE_Item': list(st.session_state.selected_ppe.values()),
            'Class_ID': list(st.session_state.selected_ppe.keys())
        }
        pd.DataFrame(config_data).to_excel(writer, sheet_name='Configuration', index=False)
    
    st.download_button(
        "📥 Download Excel Report",
        output.getvalue(),
        f"ppe_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
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
    ├── app.py                    # This Streamlit app
    ├── requirements.txt          # Dependencies list
    ├── best.pt                  # Your trained model
    └── README.md                # Optional description
    ```
    
    **Step 2: Create requirements.txt**
    ```txt
    streamlit
    opencv-python
    ultralytics
    pandas
    numpy
    pillow
    plotly
    requests
    ```
    
    **Step 3: Deploy to Streamlit Cloud**
    1. Go to [share.streamlit.io](https://share.streamlit.io)
    2. Sign in with GitHub
    3. Connect your repository
    4. Set main file path to `app.py`
    5. Click "Deploy"
    """)
    
    st.subheader("3. Configuration for Online Access")
    
    with st.expander("🔧 Advanced Configuration"):
        st.markdown("""
        **For IP Camera Access:**
        - Ensure your IP cameras are accessible from the internet
        - Use public IP addresses or domain names
        - Configure port forwarding on your router if needed
        
        **Security Considerations:**
        - Use strong passwords for IP cameras
        - Consider VPN for secure access
        - Regular security updates
        """)
    
    st.subheader("4. Access from Anywhere")
    
    st.markdown("""
    Once deployed, your app will be available at:
    `https://your-app-name.streamlit.app`
    
    **Share this URL** with anyone who needs access!
    
    **Mobile Access:**
    - The app works on mobile browsers
    - No app installation required
    - Responsive design for all devices
    """)
    
    st.subheader("5. Troubleshooting")
    
    st.markdown("""
    **Common Issues:**
    - **Model too large**: Streamlit Cloud has 1GB limit
    - **Camera not working**: IP cameras must be publicly accessible
    - **Slow performance**: Use smaller model or reduce frame rate
    
    **Solutions:**
    - Compress your model file
    - Use public IP cameras for testing
    - Optimize detection settings
    """)

if __name__ == "__main__":
    main()
