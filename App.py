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
import av
import queue
import threading
import json

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
    layout="wide",
    initial_sidebar_state="expanded"
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
    .mobile-feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
    }
    .violation-counter {
        background-color: #ff4444;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .compliance-counter {
        background-color: #44ff44;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .camera-frame {
        border: 3px solid #8B4513;
        border-radius: 10px;
        padding: 5px;
        background-color: #000;
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
    st.session_state.selected_ppe = {}  # Changed to dict with class_id: {"name": str, "color": tuple, "required": bool}
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
if 'mobile_camera_active' not in st.session_state:
    st.session_state.mobile_camera_active = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Live Monitoring"
if 'violation_count' not in st.session_state:
    st.session_state.violation_count = 0
if 'compliance_count' not in st.session_state:
    st.session_state.compliance_count = 0
if 'live_streaming' not in st.session_state:
    st.session_state.live_streaming = False
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = None
if 'capture' not in st.session_state:
    st.session_state.capture = None

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
            "model/best.pt",
            "ppe-detection.pt",  # Added common PPE model name
            "yolov8n-ppe.pt"
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
            
            # Default PPE classes and colors if they exist in the model
            ppe_colors = {
                'helmet': (0, 255, 0),      # Green for helmet
                'vest': (255, 0, 0),        # Blue for vest
                'person': (255, 255, 0),    # Yellow for person
                'gloves': (0, 255, 255),    # Cyan for gloves
                'boots': (255, 0, 255),     # Magenta for boots
                'goggles': (0, 165, 255),   # Orange for goggles
                'mask': (128, 0, 128),      # Purple for mask
                'ear_protection': (255, 192, 203),  # Pink for ear protection
                'safety': (255, 140, 0),    # Dark orange for general safety
                'uniform': (0, 100, 0)      # Dark green for uniform
            }
            
            # Auto-select common PPE classes
            auto_selected = {}
            for class_id, class_name in available_classes.items():
                class_name_lower = class_name.lower()
                for ppe_key, color in ppe_colors.items():
                    if ppe_key in class_name_lower:
                        auto_selected[class_id] = {
                            "name": class_name,
                            "color": color,
                            "required": True  # Default to required
                        }
                        break
            
            st.session_state.selected_ppe = auto_selected
            
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
            
            if available_classes:
                st.success(f"✅ Loaded {len(available_classes)} detection classes")
                if st.session_state.selected_ppe:
                    st.info(f"✅ Auto-selected {len(st.session_state.selected_ppe)} PPE classes")

def draw_detection_boxes(image, results, selected_ppe_classes):
    """Draw custom bounding boxes with different colors for detected and missing PPE"""
    if not results or len(results) == 0:
        return image, set(), []
    
    # Get detected classes in this frame
    detected_classes = set()
    
    # First pass: collect all detections
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        bbox = box.xyxy[0].cpu().numpy()
        
        # Check if this is a selected PPE class
        if class_id in selected_ppe_classes:
            detected_classes.add(class_id)
            
            # Get PPE info
            ppe_info = selected_ppe_classes[class_id]
            color = ppe_info.get('color', (0, 255, 0))
            name = ppe_info.get('name', f'Class {class_id}')
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            
            # Draw label with confidence
            label = f"{name}: {confidence:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(image, (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Check for missing required PPE
    missing_ppe = []
    for class_id, ppe_info in selected_ppe_classes.items():
        if ppe_info.get('required', True) and class_id not in detected_classes:
            missing_ppe.append(ppe_info['name'])
    
    # Draw missing PPE indicator on image
    if missing_ppe:
        # Draw red warning box at top
        warning_text = f"MISSING: {', '.join(missing_ppe[:3])}"
        if len(missing_ppe) > 3:
            warning_text += f" (+{len(missing_ppe)-3})"
        
        (text_width, text_height), baseline = cv2.getTextSize(
            warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        cv2.rectangle(image, (10, 10), (20 + text_width, 50), (0, 0, 255), -1)
        cv2.putText(image, warning_text, (15, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return image, detected_classes, missing_ppe

def process_frame_for_detection(frame):
    """Process a single frame for PPE detection"""
    if not st.session_state.model_loaded or not st.session_state.model:
        return frame, [], [], 0
    
    confidence = st.session_state.detection_settings.get('confidence', 0.5)
    selected_classes = list(st.session_state.selected_ppe.keys())
    speed_params = st.session_state.detection_settings.get('speed_params', {'imgsz': 640, 'half': False})
    
    # Run detection
    try:
        results = st.session_state.model(
            frame, 
            conf=confidence,
            classes=selected_classes,
            verbose=False,
            **speed_params
        )
        
        # Draw custom bounding boxes
        annotated_frame, detected_classes, missing_ppe = draw_detection_boxes(
            frame.copy(), results, st.session_state.selected_ppe
        )
        
        num_detections = len(results[0].boxes) if results and len(results) > 0 else 0
    except Exception as e:
        # Fallback if detection fails
        annotated_frame = frame.copy()
        detected_classes = set()
        missing_ppe = []
        num_detections = 0
    
    # Add performance overlay
    cv2.putText(annotated_frame, f"SafetyEagle AI - PPE Monitoring", (10, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Add violation/compliance counters
    violation_text = f"Violations: {st.session_state.violation_count}"
    compliance_text = f"Compliant: {st.session_state.compliance_count}"
    
    cv2.putText(annotated_frame, violation_text, (10, 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(annotated_frame, compliance_text, (10, 130), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Add detected classes info
    if detected_classes:
        detected_names = [st.session_state.selected_ppe[cid]['name'] for cid in detected_classes]
        detected_text = f"Detected: {', '.join(detected_names[:2])}"
        if len(detected_names) > 2:
            detected_text += f"... (+{len(detected_names)-2})"
        cv2.putText(annotated_frame, detected_text, (10, 160), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(annotated_frame, timestamp, (annotated_frame.shape[1] - 200, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return annotated_frame, detected_classes, missing_ppe, num_detections

def update_violation_count(missing_ppe):
    """Update violation and compliance counts"""
    if missing_ppe:
        st.session_state.violation_count += 1
        
        # Save violation record
        violation_record = {
            'timestamp': datetime.now(),
            'missing_ppe': missing_ppe,
            'selected_ppe': [ppe['name'] for ppe in st.session_state.selected_ppe.values()],
            'violation_count': st.session_state.violation_count
        }
        st.session_state.violations.append(violation_record)
        return False
    else:
        st.session_state.compliance_count += 1
        return True

def process_mobile_stream(frame):
    """Process mobile camera stream frames"""
    if frame is None:
        return None
    
    # Process frame for PPE detection
    annotated_frame, detected_classes, missing_ppe, num_detections = process_frame_for_detection(frame)
    
    # Update counts
    is_compliant = update_violation_count(missing_ppe)
    
    # Add compliance status
    status_color = (0, 255, 0) if is_compliant else (0, 0, 255)
    status_text = "COMPLIANT" if is_compliant else "VIOLATION"
    cv2.putText(annotated_frame, status_text, (annotated_frame.shape[1] - 150, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    return annotated_frame

def start_camera_capture():
    """Start camera capture"""
    if st.session_state.capture is None:
        try:
            st.session_state.capture = cv2.VideoCapture(0)
            # Set camera properties for better performance
            st.session_state.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            st.session_state.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            st.session_state.capture.set(cv2.CAP_PROP_FPS, 30)
            return True
        except Exception as e:
            st.error(f"❌ Failed to start camera: {e}")
            return False
    return True

def stop_camera_capture():
    """Stop camera capture"""
    if st.session_state.capture is not None:
        st.session_state.capture.release()
        st.session_state.capture = None

def main():
    # SafetyEagle Header
    st.markdown('<h1 class="eagle-header">🦅 SafetyEagle AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="eagle-tagline">Soaring Above Safety Standards - PPE Monitoring</p>', unsafe_allow_html=True)
    
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
        
        # Show PPE detection status
        st.sidebar.markdown("### 📊 PPE Detection Status")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Violations", st.session_state.violation_count)
        with col2:
            st.metric("Compliant", st.session_state.compliance_count)
        
        # Show selected PPE classes
        if st.session_state.selected_ppe:
            with st.sidebar.expander("🎯 Selected PPE Classes"):
                for class_id, ppe_info in list(st.session_state.selected_ppe.items())[:6]:
                    color = ppe_info.get('color', (0, 0, 0))
                    color_html = f"rgb({color[0]}, {color[1]}, {color[2]})"
                    required = "✓" if ppe_info.get('required', True) else "○"
                    st.markdown(
                        f'<span style="color:{color_html}; font-weight:bold;">■</span> '
                        f'**{ppe_info["name"]}** {required}',
                        unsafe_allow_html=True
                    )
                if len(st.session_state.selected_ppe) > 6:
                    st.write(f"... and {len(st.session_state.selected_ppe) - 6} more")
    else:
        st.sidebar.error("❌ Model Not Loaded")
        st.sidebar.info("Using simulation mode")
    
    # Sidebar navigation
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧭 Navigation")
    
    page = st.sidebar.radio("Go to", 
        ["Live Monitoring", "Mobile Live Stream", "Class Selection", "Camera Setup", "Settings", "Dashboard", "Reports", "Deployment Guide"])
    
    st.session_state.current_page = page
    
    # Show the appropriate page based on selection
    if page == "Class Selection":
        show_class_selection()
    elif page == "Camera Setup":
        show_camera_setup()
    elif page == "Mobile Live Stream":
        show_mobile_live_stream()
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

def show_mobile_live_stream():
    st.markdown('<h2 class="section-header">📱 Mobile Live Stream Analysis</h2>', unsafe_allow_html=True)
    
    if not st.session_state.model_loaded:
        st.error("❌ Please wait for model initialization on the Class Selection page first.")
        return
    
    if not st.session_state.selected_ppe:
        st.warning("⚠️ Please select PPE classes to monitor first on the Class Selection page!")
        return
    
    # Mobile live streaming introduction
    st.markdown("""
    <div class='mobile-feature-card'>
        <h3 style='color: white; margin: 0;'>📱 Live PPE Monitoring</h3>
        <p style='color: white; margin: 0.5rem 0 0 0;'>Real-time safety compliance analysis using your webcam/mobile camera</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("""
    **📱 Live Stream Instructions:**
    1. Click "Start Live Stream" below
    2. Allow camera permissions when prompted by your browser
    3. Point your camera at the work area/person
    4. SafetyEagle AI will analyze PPE compliance in real-time
    5. **Green boxes**: Detected PPE items
    6. **Red warning**: Missing required PPE
    7. Counters update in real-time
    """)
    
    # Display current PPE selection
    st.subheader("🎯 Currently Monitoring")
    if st.session_state.selected_ppe:
        cols = st.columns(min(4, len(st.session_state.selected_ppe)))
        for idx, (class_id, ppe_info) in enumerate(list(st.session_state.selected_ppe.items())[:8]):
            with cols[idx % len(cols)]:
                color = ppe_info.get('color', (0, 255, 0))
                st.markdown(f"""
                <div style="border-left: 4px solid rgb({color[0]}, {color[1]}, {color[2]});
                          padding: 0.5rem; margin: 0.25rem 0; background: #f8f9fa;">
                  <strong>{ppe_info['name']}</strong><br>
                  <small>{"Required" if ppe_info.get('required', True) else "Optional"}</small>
                </div>
                """, unsafe_allow_html=True)
    
    # Live stream controls
    st.subheader("📹 Live Stream Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("▶️ Start Live Stream", type="primary", use_container_width=True):
            st.session_state.live_streaming = True
            if start_camera_capture():
                st.success("✅ Camera started successfully!")
            st.rerun()
    
    with col2:
        if st.button("⏹️ Stop Stream", use_container_width=True):
            st.session_state.live_streaming = False
            stop_camera_capture()
            st.rerun()
    
    with col3:
        if st.button("🔄 Reset Counters", use_container_width=True):
            st.session_state.violation_count = 0
            st.session_state.compliance_count = 0
            st.session_state.violations = []
            st.rerun()
    
    # Live stream display
    if st.session_state.live_streaming:
        st.subheader("🎥 Live PPE Monitoring")
        
        # Create placeholders
        frame_placeholder = st.empty()
        stats_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Start camera if not already started
        if st.session_state.capture is None:
            if not start_camera_capture():
                st.error("❌ Failed to access camera. Please check permissions.")
                st.session_state.live_streaming = False
                return
        
        # Stream processing loop
        while st.session_state.live_streaming:
            try:
                if st.session_state.capture is None:
                    break
                    
                # Read frame from camera
                ret, frame = st.session_state.capture.read()
                
                if not ret:
                    status_placeholder.error("❌ Failed to capture frame from camera")
                    time.sleep(0.1)
                    continue
                
                # Process frame
                processed_frame = process_mobile_stream(frame)
                
                if processed_frame is not None:
                    # Convert BGR to RGB for display
                    rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    
                    # Display frame with styling
                    frame_placeholder.markdown(
                        f'<div class="camera-frame">',
                        unsafe_allow_html=True
                    )
                    frame_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
                    frame_placeholder.markdown('</div>', unsafe_allow_html=True)
                
                # Update stats
                with stats_placeholder.container():
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(f'<div class="violation-counter">{st.session_state.violation_count}</div>', unsafe_allow_html=True)
                        st.caption("Safety Violations")
                    with col2:
                        st.markdown(f'<div class="compliance-counter">{st.session_state.compliance_count}</div>', unsafe_allow_html=True)
                        st.caption("Compliant Frames")
                    with col3:
                        if st.session_state.violations:
                            last_violation = st.session_state.violations[-1]['timestamp'].strftime("%H:%M:%S")
                            st.metric("Last Violation", last_violation)
                        else:
                            st.metric("Last Violation", "None")
                    with col4:
                        total_frames = st.session_state.violation_count + st.session_state.compliance_count
                        compliance_rate = 0
                        if total_frames > 0:
                            compliance_rate = (st.session_state.compliance_count / total_frames) * 100
                        st.metric("Compliance Rate", f"{compliance_rate:.1f}%")
                
                # Show current status
                if st.session_state.violations:
                    last_missing = st.session_state.violations[-1]['missing_ppe']
                    if last_missing:
                        status_placeholder.warning(f"🚨 **Last Violation:** Missing {', '.join(last_missing)}")
                    else:
                        status_placeholder.success("✅ **Current Status:** All PPE compliant")
                else:
                    status_placeholder.info("ℹ️ **Status:** No violations yet")
                
                # Small delay to prevent overwhelming
                time.sleep(0.03)
                
            except Exception as e:
                status_placeholder.error(f"❌ Error in stream processing: {str(e)}")
                time.sleep(1)
                continue
        
        # Clean up when streaming stops
        stop_camera_capture()
        
    else:
        # Show static instructions when not streaming
        st.subheader("🎯 Detection Preview")
        
        # Create a sample detection image
        sample_img = np.ones((400, 600, 3), dtype=np.uint8) * 50
        
        # Draw sample detections like in your example
        # Person
        cv2.rectangle(sample_img, (200, 100), (400, 350), (255, 255, 0), 3)
        cv2.putText(sample_img, "Person 0.74", (210, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Helmet (green)
        cv2.rectangle(sample_img, (220, 80), (280, 100), (0, 255, 0), 3)
        cv2.putText(sample_img, "helmet 0.86", (225, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Vest (blue)
        cv2.rectangle(sample_img, (250, 150), (350, 250), (255, 0, 0), 3)
        cv2.putText(sample_img, "vest 0.76", (255, 145), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Add title and legend
        cv2.putText(sample_img, "PPE Detection Preview", (150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add legend
        cv2.rectangle(sample_img, (20, 380), (30, 390), (255, 255, 0), -1)  # Person
        cv2.putText(sample_img, "Person", (40, 390), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.rectangle(sample_img, (120, 380), (130, 390), (0, 255, 0), -1)  # Helmet
        cv2.putText(sample_img, "Helmet", (140, 390), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.rectangle(sample_img, (220, 380), (230, 390), (255, 0, 0), -1)  # Vest
        cv2.putText(sample_img, "Safety Vest", (240, 390), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        st.image(sample_img, caption="PPE Detection Preview (Similar to your example)", use_column_width=True)
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📸 Single Image Analysis", use_container_width=True):
                st.session_state.current_page = "Live Monitoring"
                st.rerun()
        with col2:
            if st.button("🎯 Class Selection", use_container_width=True):
                st.session_state.current_page = "Class Selection"
                st.rerun()
        with col3:
            if st.button("📊 View Dashboard", use_container_width=True):
                st.session_state.current_page = "Dashboard"
                st.rerun()

def show_live_monitoring():
    st.markdown('<h2 class="section-header">📹 Live Safety Monitoring</h2>', unsafe_allow_html=True)
    
    if not st.session_state.model_loaded:
        st.error("❌ Please wait for model initialization on the Class Selection page first.")
        return
    
    if not st.session_state.selected_ppe:
        st.warning("⚠️ Please select PPE classes to monitor first on the Class Selection page!")
        return
    
    # Display current configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Monitoring Controls")
        
        # Camera options
        camera_mode = st.radio(
            "Select Input Source:",
            ["Webcam", "IP Camera", "Test Mode", "Upload Image/Video"],
            help="Choose your video source"
        )
        
        # Webcam selection
        if camera_mode == "Webcam":
            st.info("Use your webcam for live PPE monitoring")
            if st.button("📷 Start Webcam Monitoring", type="primary"):
                st.session_state.current_page = "Mobile Live Stream"
                st.rerun()
        
        # IP Camera selection
        elif camera_mode == "IP Camera":
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
        
        # Upload image/video for analysis
        elif camera_mode == "Upload Image/Video":
            uploaded_file = st.file_uploader(
                "Upload image or video for PPE analysis",
                type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'],
                help="Upload an image or video to analyze for PPE compliance"
            )
            
            if uploaded_file:
                # Determine file type
                file_ext = uploaded_file.name.split('.')[-1].lower()
                
                if file_ext in ['jpg', 'jpeg', 'png']:
                    # Process image
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    if image is not None:
                        # Process single image
                        annotated_frame, detected_classes, missing_ppe, num_detections = process_frame_for_detection(image)
                        is_compliant = update_violation_count(missing_ppe)
                        
                        # Display results
                        col_img1, col_img2 = st.columns(2)
                        with col_img1:
                            st.subheader("Original Image")
                            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
                        with col_img2:
                            st.subheader("PPE Analysis")
                            st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_column_width=True)
                        
                        # Show compliance status
                        if missing_ppe:
                            st.error(f"🚨 **SAFETY VIOLATION DETECTED!**")
                            st.warning(f"**Missing PPE:** {', '.join(missing_ppe)}")
                        else:
                            st.success(f"✅ **SAFETY COMPLIANCE VERIFIED**")
                            if detected_classes:
                                detected_names = [st.session_state.selected_ppe[cid]['name'] for cid in detected_classes]
                                st.info(f"**Detected PPE:** {', '.join(detected_names)}")
                            else:
                                st.info("No PPE detected in the image")
                
                elif file_ext in ['mp4', 'avi', 'mov']:
                    # Process video
                    st.info("Video processing selected. This may take a moment...")
                    if st.button("🎬 Process Video", type="primary"):
                        process_uploaded_video(uploaded_file)
        
        # Test mode
        elif camera_mode == "Test Mode":
            st.info("Test mode with simulated PPE scenarios")
            if st.button("🧪 Start Test Mode", type="primary"):
                st.session_state.monitoring = True
                start_test_mode()
            
            if st.button("⏹️ Stop Test Mode"):
                st.session_state.monitoring = False
                st.rerun()
        
        # Performance info
        st.info(f"""
        **Current Settings:**
        - Monitoring: {len(st.session_state.selected_ppe)} PPE classes
        - Confidence: {st.session_state.detection_settings.get('confidence', 0.5)}
        - Speed: {st.session_state.detection_settings.get('speed', 'medium').title()}
        - Frame Skip: {st.session_state.detection_settings.get('frame_skip', 3)}
        """)
    
    with col2:
        st.subheader("Live Stats")
        st.metric("Violations Detected", st.session_state.violation_count)
        st.metric("Compliant Frames", st.session_state.compliance_count)
        st.metric("Monitoring Status", "ACTIVE" if st.session_state.monitoring else "INACTIVE")
        st.metric("Selected PPE Classes", len(st.session_state.selected_ppe))
        
        # Current PPE status
        st.subheader("PPE Status")
        if st.session_state.selected_ppe:
            for class_id, ppe_info in list(st.session_state.selected_ppe.items())[:3]:
                color = ppe_info.get('color', (0, 0, 0))
                required = "✓" if ppe_info.get('required', True) else "○"
                st.markdown(
                    f'<span style="color:rgb({color[0]},{color[1]},{color[2]}); font-weight:bold;">■</span> '
                    f'{ppe_info["name"]} {required}',
                    unsafe_allow_html=True
                )
            if len(st.session_state.selected_ppe) > 3:
                st.write(f"... +{len(st.session_state.selected_ppe) - 3} more")
        
        # Quick navigation
        st.subheader("Quick Navigation")
        if st.button("📱 Mobile Live Stream", use_container_width=True):
            st.session_state.current_page = "Mobile Live Stream"
            st.rerun()
        if st.button("🎯 Class Selection", use_container_width=True):
            st.session_state.current_page = "Class Selection"
            st.rerun()
        if st.button("📊 Dashboard", use_container_width=True):
            st.session_state.current_page = "Dashboard"
            st.rerun()

def show_class_selection():
    st.markdown('<h2 class="section-header">🎯 Select PPE Detection Classes</h2>', unsafe_allow_html=True)
    
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
    
    # PPE color mapping
    ppe_colors = {
        'helmet': (0, 255, 0),      # Green
        'vest': (255, 0, 0),        # Blue
        'person': (255, 255, 0),    # Yellow
        'gloves': (0, 255, 255),    # Cyan
        'boots': (255, 0, 255),     # Magenta
        'goggles': (0, 165, 255),   # Orange
        'mask': (128, 0, 128),      # Purple
        'ear_protection': (255, 192, 203),  # Pink
        'safety': (255, 140, 0),    # Dark orange
        'uniform': (0, 100, 0)      # Dark green
    }
    
    # Display all available classes for selection
    st.subheader("Available Detection Classes")
    st.info("Select PPE classes to monitor and set their colors:")
    
    # Create columns for better organization
    num_columns = 3
    classes_list = list(st.session_state.available_classes.items())
    classes_per_column = (len(classes_list) + num_columns - 1) // num_columns
    
    cols = st.columns(num_columns)
    
    selected_classes = st.session_state.selected_ppe.copy()
    
    for i, (class_id, class_name) in enumerate(classes_list):
        col_idx = i // classes_per_column
        with cols[col_idx]:
            # Determine default color based on class name
            default_color = (0, 255, 0)  # Default green
            class_name_lower = class_name.lower()
            for ppe_key, color in ppe_colors.items():
                if ppe_key in class_name_lower:
                    default_color = color
                    break
            
            # Checkbox for selection
            is_selected = st.checkbox(
                f"**{class_name}** (ID: {class_id})",
                value=class_id in selected_classes,
                key=f"class_{class_id}"
            )
            
            if is_selected:
                # Color picker
                if class_id in selected_classes:
                    current_color = selected_classes[class_id].get('color', default_color)
                else:
                    current_color = default_color
                
                # Convert BGR to RGB for color picker
                color_rgb = (current_color[2], current_color[1], current_color[0])
                selected_color = st.color_picker(
                    "Color",
                    value=f"#{color_rgb[0]:02x}{color_rgb[1]:02x}{color_rgb[2]:02x}",
                    key=f"color_{class_id}"
                )
                
                # Convert hex to BGR
                hex_color = selected_color.lstrip('#')
                rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
                
                # Required checkbox
                is_required = st.checkbox(
                    "Required",
                    value=selected_classes.get(class_id, {}).get('required', True),
                    key=f"required_{class_id}"
                )
                
                selected_classes[class_id] = {
                    "name": class_name,
                    "color": bgr_color,
                    "required": is_required
                }
            elif class_id in selected_classes:
                del selected_classes[class_id]
    
    # Selection actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ Select All PPE Classes", use_container_width=True):
            # Auto-select classes that look like PPE
            auto_selected = {}
            for class_id, class_name in st.session_state.available_classes.items():
                class_name_lower = class_name.lower()
                is_ppe = False
                color = (0, 255, 0)  # Default green
                
                for ppe_key, ppe_color in ppe_colors.items():
                    if ppe_key in class_name_lower:
                        is_ppe = True
                        color = ppe_color
                        break
                
                if is_ppe:
                    auto_selected[class_id] = {
                        "name": class_name,
                        "color": color,
                        "required": True
                    }
            
            st.session_state.selected_ppe = auto_selected
            st.success(f"✅ Selected {len(auto_selected)} PPE classes!")
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
            st.rerun()
    
    # Show current selection summary
    if selected_classes:
        st.subheader("Current Selection Summary")
        st.info(f"**Selected {len(selected_classes)} out of {len(st.session_state.available_classes)} classes:**")
        
        # Display selected classes with their colors
        st.write("**Selected PPE Classes:**")
        num_cols = min(4, len(selected_classes))
        cols = st.columns(num_cols)
        for idx, (class_id, ppe_info) in enumerate(selected_classes.items()):
            with cols[idx % num_cols]:
                color = ppe_info['color']
                required = "✓" if ppe_info.get('required', True) else "○"
                st.markdown(
                    f'<div style="border-left: 4px solid rgb({color[0]}, {color[1]}, {color[2]}); '
                    f'padding: 0.5rem; margin: 0.25rem 0; background: #f8f9fa;">'
                    f'<strong>{ppe_info["name"]}</strong><br>'
                    f'<small>ID: {class_id} | Required: {required}</small>'
                    f'</div>',
                    unsafe_allow_html=True
                )
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
            step=0.05,
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
    st.subheader("Currently Selected PPE Classes")
    if st.session_state.selected_ppe:
        selected_classes_display = [ppe_info["name"] for ppe_info in st.session_state.selected_ppe.values()]
        st.info(f"**Monitoring {len(selected_classes_display)} PPE classes:** {', '.join(selected_classes_display[:5])}{'...' if len(selected_classes_display) > 5 else ''}")
        
        # Show required vs optional
        required_count = sum(1 for ppe in st.session_state.selected_ppe.values() if ppe.get('required', True))
        optional_count = len(st.session_state.selected_ppe) - required_count
        st.write(f"**Required:** {required_count} | **Optional:** {optional_count}")

def start_ip_camera_monitoring(camera_url):
    """IP camera monitoring with PPE detection"""
    if not CV2_AVAILABLE:
        st.error("❌ OpenCV not available for camera streaming")
        return
        
    st.info(f"🌐 Connecting to IP camera: {camera_url}")
    
    confidence = st.session_state.detection_settings.get('confidence', 0.5)
    frame_skip = st.session_state.detection_settings.get('frame_skip', 3)
    speed_params = st.session_state.detection_settings.get('speed_params', {'imgsz': 640, 'half': False})
    
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
    
    # Run monitoring loop
    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    frame_count = 0
    start_time = time.time()
    
    while st.session_state.monitoring and cap.isOpened():
        try:
            ret, frame = cap.read()
            
            if not ret:
                st.error("❌ Failed to read frame from camera")
                break
            
            frame_count += 1
            
            if frame_count % frame_skip == 0:
                # Process frame for PPE detection
                annotated_frame, detected_classes, missing_ppe, num_detections = process_frame_for_detection(frame)
                
                # Update violation counts
                is_compliant = update_violation_count(missing_ppe)
                
                # Display frame
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(annotated_frame_rgb, caption="Live PPE Monitoring", use_column_width=True)
                
                # Update status
                if missing_ppe:
                    status_placeholder.warning(f"🚨 Missing PPE: {', '.join(missing_ppe)}")
                else:
                    if detected_classes:
                        detected_names = [st.session_state.selected_ppe[cid]['name'] for cid in detected_classes]
                        status_placeholder.success(f"✅ PPE Compliant: {', '.join(detected_names)}")
                    else:
                        status_placeholder.info("ℹ️ No PPE detected in frame")
            
            # Update stats every second
            if frame_count % 30 == 0:
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                
                with stats_placeholder.container():
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Violations", st.session_state.violation_count)
                    with col2:
                        st.metric("Compliant", st.session_state.compliance_count)
                    with col3:
                        st.metric("FPS", f"{fps:.1f}")
            
            time.sleep(0.01)
            
        except Exception as e:
            st.error(f"Monitoring error: {e}")
            break
    
    cap.release()

def start_test_mode():
    """Test mode with PPE detection simulation"""
    st.success("🎯 SafetyEagle Test Mode Active - PPE Detection Simulation")
    
    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    frame_count = 0
    
    while st.session_state.monitoring:
        try:
            # Create test image
            test_image = np.ones((480, 640, 3), dtype=np.uint8) * 50
            
            # Draw simulated person
            cv2.rectangle(test_image, (200, 100), (440, 400), (255, 255, 0), 2)
            cv2.putText(test_image, "Person 0.74", (210, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Simulate different PPE scenarios
            scenario = (frame_count // 60) % 4
            
            missing_ppe = []
            if scenario == 0:
                # All PPE present
                cv2.rectangle(test_image, (250, 80), (300, 100), (0, 255, 0), 2)  # Helmet
                cv2.putText(test_image, "helmet 0.86", (255, 75), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.rectangle(test_image, (280, 150), (360, 250), (255, 0, 0), 2)  # Vest
                cv2.putText(test_image, "vest 0.76", (285, 145), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                status = "All PPE Present"
                
            elif scenario == 1:
                # Missing helmet
                cv2.rectangle(test_image, (280, 150), (360, 250), (255, 0, 0), 2)  # Vest only
                cv2.putText(test_image, "vest 0.71", (285, 145), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                missing_ppe = ["Helmet"]
                status = "Missing Helmet"
                
            elif scenario == 2:
                # Missing vest
                cv2.rectangle(test_image, (250, 80), (300, 100), (0, 255, 0), 2)  # Helmet only
                cv2.putText(test_image, "helmet 0.90", (255, 75), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                missing_ppe = ["Vest"]
                status = "Missing Vest"
                
            else:
                # No PPE
                missing_ppe = ["Helmet", "Vest"]
                status = "No PPE Detected"
            
            # Add title and status
            cv2.putText(test_image, "SAFETYEAGLE TEST MODE", (150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(test_image, f"Status: {status}", (150, 450), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add counters
            cv2.putText(test_image, f"Violations: {st.session_state.violation_count}", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(test_image, f"Compliant: {st.session_state.compliance_count}", (10, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Update violation counts
            if missing_ppe:
                st.session_state.violation_count += 1
                status_placeholder.warning(f"🚨 Test Violation: Missing {', '.join(missing_ppe)}")
            else:
                st.session_state.compliance_count += 1
                status_placeholder.success("✅ Test: All PPE Present")
            
            # Display frame
            test_image_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(test_image_rgb, caption="SafetyEagle Test Mode", use_column_width=True)
            
            # Update stats
            with stats_placeholder.container():
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Test Violations", st.session_state.violation_count)
                with col2:
                    st.metric("Test Compliant", st.session_state.compliance_count)
            
            frame_count += 1
            time.sleep(1.0)  # Slower for test mode
            
        except Exception as e:
            st.error(f"Test mode error: {e}")
            break

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
    stats_placeholder = st.empty()
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        st.warning("⚠️ Could not determine video length. Processing with unknown duration.")
        total_frames = 1000  # Default estimate
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        annotated_frame, detected_classes, missing_ppe, num_detections = process_frame_for_detection(frame)
        
        # Update counts
        is_compliant = update_violation_count(missing_ppe)
        
        # Display frame
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(annotated_frame_rgb, caption="Video Analysis - PPE Detection", use_column_width=True)
        
        # Update status
        if missing_ppe:
            status_placeholder.warning(f"Violations: {', '.join(missing_ppe)}")
        else:
            status_placeholder.info("No violations detected")
        
        # Update stats
        with stats_placeholder.container():
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Frame", frame_count)
            with col2:
                if detected_classes:
                    detected_names = [st.session_state.selected_ppe[cid]['name'] for cid in detected_classes]
                    st.metric("Detected", ', '.join(detected_names[:2]))
        
        frame_count += 1
        if total_frames > 0:
            progress_bar.progress(min(frame_count / total_frames, 1.0))
        time.sleep(0.03)
    
    cap.release()
    try:
        os.unlink(video_path)
    except:
        pass
    st.success("✅ Video analysis completed!")

def show_dashboard():
    st.markdown('<h2 class="section-header">📊 Safety Dashboard</h2>', unsafe_allow_html=True)
    
    # Display violation and compliance stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Violations", st.session_state.violation_count)
    
    with col2:
        st.metric("Compliant Frames", st.session_state.compliance_count)
    
    with col3:
        compliance_rate = 0
        total = st.session_state.violation_count + st.session_state.compliance_count
        if total > 0:
            compliance_rate = (st.session_state.compliance_count / total) * 100
        st.metric("Compliance Rate", f"{compliance_rate:.1f}%")
    
    with col4:
        if st.session_state.violations:
            last_violation = st.session_state.violations[-1]['timestamp']
            time_diff = datetime.now() - last_violation
            hours_since = time_diff.total_seconds() / 3600
            st.metric("Hours Since Last", f"{hours_since:.1f}h")
        else:
            st.metric("Hours Since Last", "No violations")
    
    # Show violation details if any
    if st.session_state.violations:
        st.subheader("Recent Violations")
        
        # Create DataFrame for analysis
        df = pd.DataFrame([
            {
                'Timestamp': v['timestamp'],
                'Missing PPE': ', '.join(v['missing_ppe']),
                'All PPE Monitored': ', '.join(v['selected_ppe']),
                'Hour': v['timestamp'].hour
            }
            for v in st.session_state.violations[-20:]  # Last 20 violations
        ])
        
        if not df.empty:
            # Show violations by hour
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Violations by Hour")
                hourly_data = df.groupby('Hour').size().reset_index(name='Count')
                fig = px.bar(
                    hourly_data,
                    x='Hour',
                    y='Count',
                    title="Safety Violations by Hour of Day",
                    color='Count',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Missing PPE Distribution")
                # Count missing PPE items
                all_missing = []
                for missing_str in df['Missing PPE']:
                    all_missing.extend([item.strip() for item in missing_str.split(',')])
                
                missing_counts = pd.Series(all_missing).value_counts()
                if not missing_counts.empty:
                    fig = px.pie(
                        values=missing_counts.values,
                        names=missing_counts.index,
                        title="Most Frequently Missing PPE",
                        hole=0.3
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No missing PPE data available")
            
            # Show recent violations table
            st.subheader("Recent Violation Details")
            st.dataframe(
                df.sort_values('Timestamp', ascending=False).head(10),
                use_container_width=True,
                column_config={
                    "Timestamp": st.column_config.DatetimeColumn(
                        "Timestamp",
                        format="YYYY-MM-DD HH:mm:ss"
                    )
                }
            )
    else:
        st.info("No safety violations recorded yet. Start monitoring to see data.")

def show_reports():
    st.markdown('<h2 class="section-header">📈 Safety Reports & Analytics</h2>', unsafe_allow_html=True)
    
    if not st.session_state.violations:
        st.warning("No safety data available for reports. Start monitoring first.")
        return
    
    st.info(f"**Current Safety Monitoring:** {len(st.session_state.selected_ppe)} PPE classes selected")
    
    if st.button("📊 Generate Excel Report", type="primary"):
        generate_excel_report()
    
    df = pd.DataFrame([
        {
            'Timestamp': v['timestamp'],
            'Missing Safety Items': v['missing_ppe'],
            'Monitored Classes': v['selected_ppe'],
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
            st.session_state.violation_count = 0
            st.session_state.compliance_count = 0
            st.rerun()

def generate_excel_report():
    """Generate Excel report"""
    df = pd.DataFrame([
        {
            'Timestamp': v['timestamp'],
            'Missing_Safety_Items': ', '.join(v['missing_ppe']),
            'Monitored_Classes': ', '.join(v['selected_ppe']),
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
            'Class_Name': [ppe['name'] for ppe in st.session_state.selected_ppe.values()],
            'Required': [ppe.get('required', True) for ppe in st.session_state.selected_ppe.values()]
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
    
    st.info("🦅 **SafetyEagle AI** - PPE Monitoring System")
    
    st.subheader("🚀 Features:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **✅ Core Features:**
        - Real-time PPE detection
        - Color-coded bounding boxes
        - Missing PPE alerts
        - Violation counting
        - Compliance tracking
        - Mobile/webcam support
        - IP camera integration
        """)
    
    with col2:
        st.markdown("""
        **🔧 Technical Stack:**
        - YOLOv8 object detection
        - OpenCV video processing
        - Streamlit web interface
        - Plotly analytics
        - Pandas data handling
        - Real-time processing
        """)
    
    with col3:
        st.markdown("""
        **📊 Analytics:**
        - Live violation counters
        - Compliance rate tracking
        - Hourly violation patterns
        - Missing PPE distribution
        - Excel/CSV reports
        - Real-time dashboards
        """)
    
    st.subheader("🎯 PPE Detection System")
    
    st.markdown("""
    **🔍 How It Works:**
    
    1. **Class Selection**: Choose which PPE items to monitor
    2. **Color Coding**: Each PPE type gets a distinct color
    3. **Real-time Detection**: Live processing of video streams
    4. **Violation Alerting**: Red warnings for missing PPE
    5. **Compliance Tracking**: Counters update in real-time
    
    **Supported PPE Types:**
    - Helmets (Green boxes)
    - Safety Vests (Blue boxes)  
    - Gloves, Boots, Goggles
    - Masks, Ear Protection
    - Safety Uniforms
    - Custom PPE items
    
    **Mobile/Webcam Features:**
    - Direct camera access
    - Real-time analysis
    - Violation counting
    - Compliance tracking
    - Instant feedback
    """)
    
    st.success("🎉 **SafetyEagle AI PPE Monitoring is now operational!**")

if __name__ == "__main__":
    main()
