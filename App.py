import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
import os
from pathlib import Path
import tempfile
from PIL import Image
import io
import cv2
import requests
import base64

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

# Custom CSS for Oil & Gas SafetyEagle branding
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
    .project-info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .tab-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    .ppe-item {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem;
        border-left: 3px solid #8B4513;
    }
    .violation-alert {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .detection-box {
        border: 2px solid #4CAF50;
        background-color: rgba(76, 175, 80, 0.1);
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
    }
    .analysis-result {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #2196F3;
    }
    .mobile-camera-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .qr-code {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .report-download-section {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
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
            'company_name': '',
            'project_name': '',
            'engineer_name': '',
            'contractor_name': '',
            'work_type': '',
            'project_hours': 0,
            'workers_assigned': 0,
            'start_date': datetime.now()
        }
    if 'captured_photos' not in st.session_state:
        st.session_state.captured_photos = []
    if 'uploaded_videos' not in st.session_state:
        st.session_state.uploaded_videos = []
    if 'current_camera_url' not in st.session_state:
        st.session_state.current_camera_url = ""
    if 'live_analysis_running' not in st.session_state:
        st.session_state.live_analysis_running = False
    if 'photo_analysis_results' not in st.session_state:
        st.session_state.photo_analysis_results = []
    if 'video_analysis_results' not in st.session_state:
        st.session_state.video_analysis_results = []
    if 'mobile_camera_active' not in st.session_state:
        st.session_state.mobile_camera_active = False
    if 'mobile_camera_url' not in st.session_state:
        st.session_state.mobile_camera_url = ""
    if 'mobile_live_feed_active' not in st.session_state:
        st.session_state.mobile_live_feed_active = False

# Standard Oil & Gas PPE classes
OIL_GAS_PPE_CLASSES = {
    0: "Hard Hat",
    1: "Safety Glasses",
    2: "High-Vis Vest",
    3: "Safety Gloves",
    4: "Safety Boots",
    5: "Hearing Protection",
    6: "Face Shield",
    7: "Respirator",
    8: "Fire Retardant Clothing",
    9: "Harness",
    10: "Gas Monitor"
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
                    break
            except Exception as e:
                continue
        
        if model is None:
            # Fallback to standard PPE classes for demo
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
            return None, OIL_GAS_PPE_CLASSES, "demo"
            
    except Exception as e:
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

def show_mobile_camera_live_feed():
    """Show live feed from mobile camera"""
    if not st.session_state.mobile_camera_active or not st.session_state.mobile_camera_url:
        return
    
    st.subheader("📱 Mobile Camera Live Feed")
    
    # Create placeholder for live feed
    feed_placeholder = st.empty()
    status_placeholder = st.empty()
    
    try:
        # For IP Webcam - try video feed endpoints
        video_endpoints = [
            "/video",
            "/videofeed", 
            "/live",
            "/stream",
            "/ipcam/video.mjpeg",
            "/ipcam/video"
        ]
        
        cap = None
        for endpoint in video_endpoints:
            try:
                video_url = st.session_state.mobile_camera_url + endpoint
                cap = cv2.VideoCapture(video_url)
                if cap.isOpened():
                    break
                else:
                    cap = None
            except:
                cap = None
        
        if cap is None:
            # Try direct URL as video feed
            cap = cv2.VideoCapture(st.session_state.mobile_camera_url)
        
        if cap and cap.isOpened():
            status_placeholder.success("✅ Mobile camera live feed active")
            
            while st.session_state.mobile_live_feed_active and cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB for display
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    feed_placeholder.image(frame_rgb, caption="Mobile Camera Live Feed", use_column_width=True)
                else:
                    status_placeholder.error("❌ Failed to read frame from mobile camera")
                    break
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.1)
            
            cap.release()
        else:
            status_placeholder.warning("⚠️ Live video feed not available. Use photo capture instead.")
            
    except Exception as e:
        status_placeholder.error(f"❌ Error in mobile camera feed: {e}")

def capture_from_mobile_camera():
    """Capture photo from mobile camera"""
    if not st.session_state.mobile_camera_url:
        st.error("❌ No mobile camera connected.")
        return None
    
    try:
        # For IP Webcam app - try different photo endpoints
        photo_endpoints = [
            "/photo.jpg",
            "/shot.jpg",
            "/capture",
            "/screenshot.jpg",
            "/image.jpg",
            "/picture.jpg"
        ]
        
        for endpoint in photo_endpoints:
            try:
                url = st.session_state.mobile_camera_url + endpoint
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    image = Image.open(io.BytesIO(response.content))
                    return np.array(image)
            except Exception as e:
                continue
        
        # If photo endpoints fail, try video capture
        try:
            cap = cv2.VideoCapture(st.session_state.mobile_camera_url)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    return frame_rgb
        except:
            pass
        
        st.error("❌ Cannot capture from mobile camera. Try different URL or check connection.")
        return None
                
    except Exception as e:
        st.error(f"❌ Error capturing from mobile camera: {e}")
        return None

def show_mobile_camera_photos():
    """Mobile camera photo capture interface"""
    st.subheader("📱 Mobile Camera Capture")
    
    # Show mobile camera connection section
    show_mobile_camera_connection()
    
    # Live feed and capture controls
    if st.session_state.mobile_camera_active:
        st.subheader("🎯 Capture Controls")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📸 Capture Photo", type="primary", use_container_width=True):
                with st.spinner("Capturing from mobile camera..."):
                    photo = capture_from_mobile_camera()
                    if photo is not None:
                        st.session_state.captured_photos.append({
                            'image': photo,
                            'timestamp': datetime.now(),
                            'source': 'Mobile Camera'
                        })
                        st.success("✅ Photo captured successfully!")
                        st.rerun()
        
        with col2:
            if st.button("📺 Toggle Live Feed", use_container_width=True):
                st.session_state.mobile_live_feed_active = not st.session_state.mobile_live_feed_active
                st.rerun()
        
        with col3:
            if st.button("🔄 Test Connection", use_container_width=True):
                test_photo = capture_from_mobile_camera()
                if test_photo is not None:
                    st.success("✅ Connection working!")
                    st.image(test_photo, caption="Test Capture", use_column_width=True)
                else:
                    st.error("❌ Connection failed")
        
        with col4:
            if st.button("🗑️ Clear Photos", use_container_width=True):
                st.session_state.captured_photos = [p for p in st.session_state.captured_photos if p['source'] != 'Mobile Camera']
                st.rerun()
        
        # Show live feed if active
        if st.session_state.mobile_live_feed_active:
            show_mobile_camera_live_feed()
        
        # Display captured photos
        mobile_photos = [p for p in st.session_state.captured_photos if p['source'] == 'Mobile Camera']
        if mobile_photos:
            st.subheader("📷 Captured Photos from Mobile")
            
            # Show last 3 mobile photos
            for i, photo_data in enumerate(mobile_photos[-3:]):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.image(photo_data['image'], 
                            caption=f"Mobile Capture: {photo_data['timestamp'].strftime('%H:%M:%S')}",
                            use_column_width=True)
                with col2:
                    col2a, col2b = st.columns(2)
                    with col2a:
                        if st.button(f"🔍 Analyze", key=f"analyze_mobile_{i}"):
                            with st.spinner("Analyzing photo..."):
                                analysis_result = analyze_image(photo_data['image'], f"Mobile Photo {i+1}")
                                if analysis_result:
                                    st.session_state.photo_analysis_results.append(analysis_result)
                                    display_analysis_results(analysis_result, "Mobile Camera Analysis")
                    with col2b:
                        if st.button(f"🗑️", key=f"delete_mobile_{i}"):
                            st.session_state.captured_photos.remove(photo_data)
                            st.rerun()
        else:
            st.info("No photos captured from mobile camera yet. Click 'Capture Photo' to get started.")
    else:
        st.info("👆 Connect a mobile camera above to start capturing photos")

def show_mobile_camera_connection():
    """Show mobile camera connection interface"""
    st.markdown('<div class="mobile-camera-card">', unsafe_allow_html=True)
    st.subheader("🔗 Mobile Camera Connection")
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Connection methods
        connection_method = st.radio(
            "Choose Connection Method:",
            ["IP Webcam App", "Custom Camera URL"]
        )
        
        if connection_method == "IP Webcam App":
            st.markdown("""
            **Using IP Webcam App:**
            1. 📱 Install 'IP Webcam' from Play Store/App Store
            2. 🌐 Open app and start server
            3. 🔗 Enter the URL shown in the app below
            4. 📸 Use the capture button to take photos
            """)
            
            default_url = st.text_input(
                "IP Webcam URL:",
                value=st.session_state.mobile_camera_url or "http://192.168.1.100:8080",
                placeholder="http://your-phone-ip:8080",
                help="Enter the URL shown in IP Webcam app"
            )
            
        else:  # Custom Camera URL
            st.markdown("""
            **Using Custom Camera:**
            1. 📱 Ensure your mobile camera app provides a stream URL
            2. 🔗 Enter the camera stream URL below
            3. 📸 Use the capture button to take photos
            """)
            
            default_url = st.text_input(
                "Camera Stream URL:",
                value=st.session_state.mobile_camera_url,
                placeholder="http://ip:port/stream or rtsp://ip:port/stream",
                help="Enter your mobile camera stream URL"
            )
        
        # Test and save connection
        col1a, col1b = st.columns(2)
        with col1a:
            if st.button("🔗 Connect Mobile Camera", type="primary", use_container_width=True):
                if default_url:
                    st.session_state.mobile_camera_url = default_url
                    st.session_state.mobile_camera_active = True
                    st.success(f"✅ Mobile camera connected: {default_url}")
                    
                    # Test connection
                    with st.spinner("Testing connection..."):
                        test_photo = capture_from_mobile_camera()
                        if test_photo is not None:
                            st.success("✅ Connection successful! Ready to capture photos.")
                            st.image(test_photo, caption="Test Capture", use_column_width=True)
                        else:
                            st.error("❌ Connection test failed. Please check URL and try different endpoints.")
                else:
                    st.error("❌ Please enter a camera URL")
        
        with col1b:
            if st.session_state.mobile_camera_active:
                if st.button("🔴 Disconnect", use_container_width=True):
                    st.session_state.mobile_camera_active = False
                    st.session_state.mobile_live_feed_active = False
                    st.session_state.mobile_camera_url = ""
                    st.rerun()
    
    with col2:
        st.subheader("📱 Quick Setup Guide")
        
        st.markdown("""
        **IP Webcam Setup:**
        1. Install **IP Webcam** from app store
        2. Open app and scroll down
        3. Tap **"Start server"**
        4. Note the URL shown (e.g., `http://192.168.1.100:8080`)
        5. Enter that URL in the left panel
        
        **Troubleshooting:**
        - 🔄 Ensure phone and computer are on same WiFi
        - 📱 Keep phone screen on during capture
        - 🔒 Disable mobile data during testing
        - 🌐 Try different URL endpoints if capture fails
        """)
        
        # Common URL patterns for testing
        st.subheader("🔧 Common Endpoints")
        st.markdown("""
        Try these URL patterns:
        - `http://[phone-ip]:8080`
        - `http://[phone-ip]:8080/video`
        - `http://[phone-ip]:8080/videofeed`
        - `http://[phone-ip]:8080/photo.jpg`
        """)

def show_photo_analysis():
    """Photo analysis tab with mobile camera support"""
    st.markdown('<h2 class="section-header">🖼️ Photo Analysis</h2>', unsafe_allow_html=True)
    
    st.info("Capture photos from mobile camera, webcam, or upload existing photos for PPE analysis")
    
    # Create tabs for different photo sources
    tab1, tab2, tab3 = st.tabs(["📱 Mobile Camera", "📹 Web Camera", "📁 Upload Photos"])
    
    with tab1:
        show_mobile_camera_photos()
    
    with tab2:
        show_web_camera_photos()
    
    with tab3:
        show_upload_photos()

def show_web_camera_photos():
    """Web camera photo capture interface"""
    st.subheader("📹 Web Camera Capture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌐 IP Camera Setup")
        camera_url = st.text_input(
            "IP Camera URL:",
            value=st.session_state.current_camera_url,
            placeholder="http://ip:port/video or rtsp://ip:port/stream",
            help="Enter your IP camera stream URL",
            key="web_camera_url"
        )
        
        if st.button("🔗 Test & Save Connection", use_container_width=True, key="test_web_cam"):
            if camera_url:
                try:
                    cap = cv2.VideoCapture(camera_url)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        cap.release()
                        if ret:
                            st.session_state.current_camera_url = camera_url
                            if camera_url not in st.session_state.camera_urls:
                                st.session_state.camera_urls.append(camera_url)
                            st.success("✅ Camera connected successfully!")
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            st.image(frame_rgb, caption="Camera Test Frame", use_column_width=True)
                        else:
                            st.error("❌ Connected but no frame received")
                    else:
                        st.error("❌ Cannot connect to camera")
                except Exception as e:
                    st.error(f"❌ Connection test failed: {e}")
            else:
                st.error("Please enter a camera URL")
    
    with col2:
        st.subheader("📸 Capture Controls")
        if st.session_state.current_camera_url:
            if st.button("📷 Capture Photo from Webcam", type="primary", use_container_width=True, key="capture_web"):
                photo = capture_photo_from_camera()
                if photo is not None:
                    st.session_state.captured_photos.append({
                        'image': photo,
                        'timestamp': datetime.now(),
                        'source': 'Web Camera'
                    })
                    st.success("✅ Photo captured successfully!")
                    st.rerun()
        else:
            st.warning("⚠️ No web camera configured")
        
        # Display web camera photos
        web_photos = [p for p in st.session_state.captured_photos if p['source'] == 'Web Camera']
        if web_photos:
            st.subheader("📹 Web Camera Photos")
            for i, photo_data in enumerate(web_photos[-2:]):
                st.image(photo_data['image'], 
                        caption=f"Webcam: {photo_data['timestamp'].strftime('%H:%M:%S')}",
                        use_column_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"🔍 Analyze Web {i+1}", key=f"analyze_web_{i}"):
                        with st.spinner("Analyzing photo..."):
                            analysis_result = analyze_image(photo_data['image'], f"Webcam Photo {i+1}")
                            if analysis_result:
                                st.session_state.photo_analysis_results.append(analysis_result)
                                display_analysis_results(analysis_result, "Web Camera Analysis")
                with col2:
                    if st.button(f"🗑️ Delete Web {i+1}", key=f"delete_web_{i}"):
                        st.session_state.captured_photos.remove(photo_data)
                        st.rerun()

def show_upload_photos():
    """Photo upload interface"""
    st.subheader("📁 Upload Photos for Analysis")
    
    uploaded_files = st.file_uploader(
        "Choose image files", 
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload photos to analyze for PPE compliance",
        accept_multiple_files=True,
        key="photo_uploader"
    )
    
    if uploaded_files:
        for i, file in enumerate(uploaded_files):
            image = Image.open(file)
            st.image(image, caption=f"Uploaded: {file.name}", use_column_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"🔍 Analyze {file.name}", key=f"analyze_upload_{i}"):
                    with st.spinner(f"Analyzing {file.name}..."):
                        analysis_result = analyze_image(image, f"Uploaded: {file.name}")
                        if analysis_result:
                            st.session_state.photo_analysis_results.append(analysis_result)
                            display_analysis_results(analysis_result, "Uploaded Photo Analysis")
            with col2:
                if st.button(f"🗑️ Remove {file.name}", key=f"remove_upload_{i}"):
                    # Note: We can't actually remove from uploaded_files, but we can rerun to clear
                    st.rerun()

def display_analysis_results(analysis_result, analysis_type):
    """Display analysis results in a structured format"""
    st.markdown(f'<div class="analysis-result">', unsafe_allow_html=True)
    st.subheader(f"🔍 {analysis_type} Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Display annotated image
        if 'annotated_image' in analysis_result:
            display_image = cv2.cvtColor(analysis_result['annotated_image'], cv2.COLOR_BGR2RGB)
            st.image(display_image, caption="Analysis Results with PPE Detections", use_column_width=True)
    
    with col2:
        # Display analysis summary
        if analysis_result['violation_count'] > 0:
            st.markdown('<div class="violation-alert">', unsafe_allow_html=True)
            st.error(f"🚨 PPE VIOLATION DETECTED!")
            st.write(f"**Missing PPE Items:** {len(analysis_result['missing_classes'])}")
            st.write(f"**Violation Count:** {analysis_result['violation_count']}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-card">', unsafe_allow_html=True)
            st.success("✅ ALL PPE COMPLIANT!")
            st.write("All required PPE items detected.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed detection information
        st.subheader("📊 Detection Details")
        st.write(f"**Detection Confidence:** {analysis_result['confidence']}")
        st.write(f"**Timestamp:** {analysis_result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"**Total Detections:** {analysis_result['total_detections']}")
        
        if analysis_result['detected_ppe']:
            st.write("**Detected PPE:**")
            for detection in analysis_result['detected_ppe']:
                st.markdown(f'<div class="detection-box">', unsafe_allow_html=True)
                st.write(f"✅ {detection['class_name']} (Confidence: {detection['confidence']:.2f})")
                st.markdown('</div>', unsafe_allow_html=True)
        
        if analysis_result['missing_classes']:
            st.write("**Missing PPE:**")
            for missing in analysis_result['missing_classes']:
                st.error(f"❌ {missing}")
    
    # Add to violations if applicable
    if analysis_result['violation_count'] > 0:
        if st.button("📝 Record Violation", type="primary", key=f"record_{analysis_type}_{datetime.now().timestamp()}"):
            if add_safety_violation(analysis_result):
                st.success("✅ Violation recorded in safety database!")
    
    # Generate and download report
    st.markdown('<div class="report-download-section">', unsafe_allow_html=True)
    st.subheader("📄 Download Analysis Report")
    
    report_content = generate_analysis_report([analysis_result], f"{analysis_type} Report")
    report_filename = f"safety_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    st.download_button(
        "📥 Download Detailed Report",
        report_content,
        report_filename,
        "text/plain",
        use_container_width=True,
        key=f"download_{analysis_type}_{datetime.now().timestamp()}"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def generate_analysis_report(analysis_data, analysis_type):
    """Generate detailed analysis report"""
    report_content = f"""
# SafetyEagle AI - {analysis_type} Report
## {st.session_state.project_info['company_name']}
### Project: {st.session_state.project_info['project_name']}

**Report Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Analysis Type:** {analysis_type}
**Safety Engineer:** {st.session_state.project_info['engineer_name']}
**Detection Confidence:** {st.session_state.detection_settings['confidence']}

## Executive Summary
- **Total Violations Detected:** {len(analysis_data)}
- **Required PPE Items:** {len(st.session_state.selected_ppe)}
- **Analysis Duration:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
- **Overall Compliance Status:** {'NON-COMPLIANT' if len(analysis_data) > 0 else 'COMPLIANT'}

## Detailed Findings
"""
    
    if analysis_data:
        for i, violation in enumerate(analysis_data, 1):
            report_content += f"""
### Violation {i}
- **Timestamp:** {violation['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
- **Missing PPE:** {', '.join(violation['missing_classes'])}
- **Detected PPE:** {', '.join([d['class_name'] for d in violation['detected_ppe']])}
- **Confidence Level:** {violation['confidence']}
- **Source:** {violation['source']}
- **Total Detections:** {violation['total_detections']}

"""
    else:
        report_content += "\nNo violations detected during analysis.\n"
    
    report_content += f"""
## Safety Recommendations
{'🚨 IMMEDIATE ACTION REQUIRED - Multiple PPE violations detected. Conduct safety briefing and retraining.' if len(analysis_data) > 5 else 
'⚠️ ENHANCED MONITORING NEEDED - Some PPE violations detected. Review safety procedures.' if len(analysis_data) > 0 else 
'✅ EXCELLENT COMPLIANCE - Continue current safety protocols.'}

## Required PPE Items
{chr(10).join([f"- {item}" for item in st.session_state.selected_ppe.values()])}

---
*Generated by SafetyEagle AI - Advanced PPE Monitoring System*
*Report Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    return report_content

def show_reports():
    """Professional reports tab with download buttons"""
    st.markdown('<h2 class="section-header">📈 Professional Safety Reports</h2>', unsafe_allow_html=True)
    
    if not st.session_state.project_info['project_name']:
        st.warning("⚠️ Please complete Project Setup first to generate professional reports.")
        return
    
    st.info("Generate comprehensive safety reports for management and regulatory compliance")
    
    # Report type selection with immediate download
    st.subheader("📋 Quick Report Generation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📊 Summary Report")
        st.markdown("High-level overview with key metrics")
        summary_report = generate_summary_report_content()
        st.download_button(
            "📥 Download Summary Report",
            summary_report,
            f"safety_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "text/plain",
            use_container_width=True
        )
    
    with col2:
        st.markdown("#### 📈 Analytics Report")
        st.markdown("Detailed analysis with violation trends")
        analytics_report = generate_analytics_report_content()
        st.download_button(
            "📥 Download Analytics Report",
            analytics_report,
            f"safety_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "text/plain",
            use_container_width=True
        )
    
    with col3:
        st.markdown("#### ⚖️ Compliance Report")
        st.markdown("Regulatory compliance documentation")
        compliance_report = generate_compliance_report_content()
        st.download_button(
            "📥 Download Compliance Report",
            compliance_report,
            f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "text/plain",
            use_container_width=True
        )
    
    # Data export section
    st.markdown("---")
    st.subheader("📁 Data Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV Export
        if st.session_state.violations:
            csv_data = pd.DataFrame(st.session_state.violations).to_csv(index=False)
            st.download_button(
                "📊 Export Violations CSV",
                csv_data,
                f"violations_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
        else:
            st.button("📊 Export Violations CSV", disabled=True, use_container_width=True, 
                     help="No violation data available")
    
    with col2:
        # Excel Export
        excel_data = generate_excel_report()
        st.download_button(
            "📈 Export Excel Report",
            excel_data,
            f"safety_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    # Custom report configuration
    st.markdown("---")
    st.subheader("🎛️ Custom Report Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        report_period = st.selectbox(
            "Report Period",
            ["Last 7 Days", "Last 30 Days", "Project to Date", "Custom Range"]
        )
        include_charts = st.checkbox("Include Charts and Graphs", value=True)
    
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
        custom_report = generate_custom_report_content(report_period, data_detail)
        st.markdown('<div class="report-download-section">', unsafe_allow_html=True)
        st.success("✅ Custom report generated successfully!")
        st.download_button(
            "📥 Download Custom Report",
            custom_report,
            f"custom_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "text/plain",
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

def generate_summary_report_content():
    """Generate summary safety report content"""
    return f"""
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

---
*Generated by SafetyEagle AI - Advanced PPE Monitoring System*
"""

def generate_analytics_report_content():
    """Generate analytics report content"""
    return f"""
# SafetyEagle AI - Safety Analytics Report
## {st.session_state.project_info['company_name']}
### Project: {st.session_state.project_info['project_name']}

**Report Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Analytics Overview
- Total Violations: {len(st.session_state.violations)}
- Daily Average: {len(st.session_state.violations) / max(1, (datetime.now().date() - st.session_state.project_info['start_date']).days):.1f}
- Compliance Trend: {calculate_compliance_rate():.1f}%

## Detailed Analysis
{generate_detailed_analysis()}

## Recommendations
{get_safety_recommendations()}
"""

def generate_compliance_report_content():
    """Generate compliance report content"""
    return f"""
# SafetyEagle AI - Compliance Report
## {st.session_state.project_info['company_name']}
### Project: {st.session_state.project_info['project_name']}

**Report Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Compliance Status
- Overall Compliance: {calculate_compliance_rate():.1f}%
- PPE Requirements Met: {len(st.session_state.selected_ppe)} items configured
- Violation Rate: {len(st.session_state.violations)} incidents

## Regulatory Checklist
- ✅ PPE Monitoring System: Active
- ✅ Violation Tracking: Implemented
- ✅ Reporting: Available
- ✅ Safety Protocols: Documented

## Required Actions
{get_safety_recommendations()}
"""

def generate_custom_report_content(period, detail):
    """Generate custom report content"""
    return f"""
# SafetyEagle AI - Custom Safety Report
## {st.session_state.project_info['company_name']}
### Project: {st.session_state.project_info['project_name']}

**Report Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Report Period:** {period}
**Detail Level:** {detail}

## Custom Analysis
This report contains customized safety analysis based on your selected parameters.

## Key Metrics
- Total Workers: {st.session_state.project_info['workers_assigned']}
- Project Duration: {st.session_state.project_info['project_hours']} hours
- Safety Violations: {len(st.session_state.violations)}
- Compliance Rate: {calculate_compliance_rate():.1f}%

## Detailed Findings
Customized analysis would appear here based on the selected period and detail level.
"""

def generate_excel_report():
    """Generate Excel report with safety data"""
    output = io.BytesIO()
    
    # Create a simple DataFrame for demonstration
    summary_data = {
        'Metric': ['Total Violations', 'Workers', 'Compliance Rate', 'Project Hours'],
        'Value': [
            len(st.session_state.violations),
            st.session_state.project_info['workers_assigned'],
            f"{calculate_compliance_rate():.1f}%",
            st.session_state.project_info['project_hours']
        ]
    }
    
    # For demonstration, we'll create a simple CSV-like structure
    excel_content = "Metric,Value\n"
    for metric, value in zip(summary_data['Metric'], summary_data['Value']):
        excel_content += f"{metric},{value}\n"
    
    return excel_content.encode()

# ... (rest of your existing functions like analyze_image, draw_detections, capture_photo_from_camera, etc.)

def main():
    # SafetyEagle Header with Oil & Gas Theme
    st.markdown('<h1 class="eagle-header">🦅 SafetyEagle AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="eagle-tagline">Advanced PPE Monitoring for Oil & Gas Industry</p>', unsafe_allow_html=True)
    
    # Initialize app on startup
    initialize_app()
    
    # Show system status in sidebar
    st.sidebar.markdown("### 🦅 SafetyEagle Status")
    st.sidebar.markdown("---")
    
    # Mobile camera status
    if st.session_state.mobile_camera_active:
        st.sidebar.success("📱 Mobile Camera Connected")
        if st.session_state.mobile_live_feed_active:
            st.sidebar.info("📺 Live Feed: Active")
        else:
            st.sidebar.info("📺 Live Feed: Inactive")
    else:
        st.sidebar.info("📱 Mobile Camera: Not Connected")
    
    # Quick stats
    st.sidebar.markdown("### 📊 Quick Stats")
    st.sidebar.metric("Total Violations", len(st.session_state.violations))
    st.sidebar.metric("Live Analysis", "Active" if st.session_state.live_analysis_running else "Inactive")
    st.sidebar.metric("Cameras Configured", len(st.session_state.camera_urls))
    
    # Main tabs for workflow
    tab_names = [
        "🏢 Project Setup", 
        "🛡️ PPE Selection", 
        "🖼️ Photo Analysis",  # Moved photo analysis up for mobile access
        "🎥 Video Analysis", 
        "📹 Live Monitoring",
        "📊 Dashboard", 
        "📈 Reports"
    ]
    
    # Create tabs
    tabs = st.tabs(tab_names)
    
    with tabs[0]:
        show_project_setup()
    with tabs[1]:
        show_ppe_selection()
    with tabs[2]:
        show_photo_analysis()  # This now includes mobile camera
    with tabs[3]:
        show_video_analysis()
    with tabs[4]:
        show_live_monitoring()
    with tabs[5]:
        show_dashboard()
    with tabs[6]:
        show_reports()  # Enhanced with download buttons

# Note: You'll need to include all the other existing functions from your previous code
# such as: show_project_setup, show_ppe_selection, show_video_analysis, show_live_monitoring, 
# show_dashboard, analyze_image, draw_detections, capture_photo_from_camera, and all helper functions.

if __name__ == "__main__":
    main()
