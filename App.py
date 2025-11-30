import streamlit as st
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import time
import os
from pathlib import Path
import tempfile
import base64
from PIL import Image
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import threading
import queue

# Set page configuration
st.set_page_config(
    page_title="PPE Safety Monitoring System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class PPEViolationMonitor:
    def __init__(self, model_path):
        try:
            self.model = YOLO(model_path)
            st.success(f"✅ Model loaded successfully: {model_path}")
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
            self.model = None
            
        self.violations_df = pd.DataFrame(columns=[
            'timestamp', 'person_id', 'missing_ppe', 'confidence', 
            'image_path', 'shift_hour', 'violation_type', 'session_id'
        ])
        self.session_start = datetime.now()
        self.violation_count = 0
        self.total_detections = 0
        
        # Define PPE requirements - UPDATE WITH YOUR ACTUAL CLASS IDs
        self.ppe_requirements = {
            'helmet': 0,    # Update with your actual class IDs
            'vest': 1,      # Update with your actual class IDs
            'gloves': 2,    # Update with your actual class IDs
        }
        
        # Create directories
        self.output_dir = Path("PPE_Monitoring_Reports")
        self.violation_images_dir = self.output_dir / "violation_images"
        self.reports_dir = self.output_dir / "reports"
        
        for directory in [self.output_dir, self.violation_images_dir, self.reports_dir]:
            directory.mkdir(exist_ok=True)

    def detect_ppe_violations(self, frame, detections):
        """Analyze detections for PPE violations"""
        violations = []
        
        if not detections or len(detections) == 0:
            return violations
            
        # Extract detected classes and their positions
        detected_classes = {}
        person_boxes = []
        
        for box in detections[0].boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy()
            
            if class_id == 0:  # Person class
                person_boxes.append(bbox)
            else:
                detected_classes[class_id] = {
                    'confidence': confidence,
                    'bbox': bbox
                }
        
        # Check each person for required PPE
        for i, person_bbox in enumerate(person_boxes):
            missing_ppe = self._check_person_ppe(person_bbox, detected_classes)
            
            if missing_ppe:
                violation = {
                    'person_id': i,
                    'missing_ppe': missing_ppe,
                    'confidence': min([detected_classes.get(cls, {}).get('confidence', 0) 
                                     for cls in self.ppe_requirements.values()]),
                    'violation_type': 'missing_ppe'
                }
                violations.append(violation)
        
        return violations

    def _check_person_ppe(self, person_bbox, detected_classes):
        """Check if a person has all required PPE"""
        missing_items = []
        
        for ppe_name, class_id in self.ppe_requirements.items():
            if class_id not in detected_classes:
                missing_items.append(ppe_name)
        
        return missing_items

    def save_violation_record(self, frame, violations, session_id):
        """Save violation records and images"""
        timestamp = datetime.now()
        
        for violation in violations:
            self.violation_count += 1
            
            # Save violation image
            img_filename = f"violation_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}_{violation['person_id']}.jpg"
            img_path = self.violation_images_dir / img_filename
            cv2.imwrite(str(img_path), frame)
            
            # Add to DataFrame
            new_record = {
                'timestamp': timestamp,
                'person_id': violation['person_id'],
                'missing_ppe': ', '.join(violation['missing_ppe']),
                'confidence': violation['confidence'],
                'image_path': str(img_path),
                'shift_hour': timestamp.hour,
                'violation_type': violation['violation_type'],
                'session_id': session_id
            }
            
            self.violations_df = pd.concat([
                self.violations_df, 
                pd.DataFrame([new_record])
            ], ignore_index=True)

    def process_frame(self, frame, session_id):
        """Process a single frame for PPE violations"""
        if self.model is None:
            return frame, []
            
        try:
            results = self.model.predict(frame, conf=0.6, verbose=False)
            violations = self.detect_ppe_violations(frame, results)
            
            if violations:
                self.save_violation_record(frame, violations, session_id)
            
            # Annotate frame
            annotated_frame = results[0].plot()
            return annotated_frame, violations
            
        except Exception as e:
            st.error(f"Error processing frame: {e}")
            return frame, []

class CameraManager:
    def __init__(self):
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=10)
        self.is_running = False
        self.thread = None
        self.current_camera_index = 0
        
    def start_camera(self, camera_index=0):
        """Start camera with multiple fallback options and settings"""
        self.current_camera_index = camera_index
        
        try:
            # Try different backend APIs
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
            
            for backend in backends:
                try:
                    self.cap = cv2.VideoCapture(camera_index, backend)
                    
                    if not self.cap.isOpened():
                        continue
                    
                    # Try different combinations of settings
                    settings_to_try = [
                        {'width': 640, 'height': 480, 'fps': 30},
                        {'width': 320, 'height': 240, 'fps': 15},
                        {'width': 1280, 'height': 720, 'fps': 10}
                    ]
                    
                    for settings in settings_to_try:
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings['width'])
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings['height'])
                        self.cap.set(cv2.CAP_PROP_FPS, settings['fps'])
                        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                        self.cap.set(cv2.CAP_PROP_EXPOSURE, 50)
                        
                        # Test if camera works
                        ret, frame = self.cap.read()
                        if ret and frame is not None and frame.size > 0:
                            st.success(f"✅ Camera connected with backend {backend} and settings {settings}")
                            self.is_running = True
                            self.thread = threading.Thread(target=self._capture_frames)
                            self.thread.daemon = True
                            self.thread.start()
                            return True
                            
                except Exception as e:
                    if self.cap:
                        self.cap.release()
                    continue
                    
        except Exception as e:
            st.error(f"Camera initialization error: {e}")
            
        st.error("❌ Could not initialize camera with any settings")
        return False
    
    def _capture_frames(self):
        """Capture frames in a separate thread"""
        consecutive_failures = 0
        max_failures = 5
        
        while self.is_running and self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                
                if ret and frame is not None and frame.size > 0:
                    consecutive_failures = 0
                    
                    # Sometimes cameras return black frames initially
                    if np.mean(frame) > 10:  # Check if frame is not completely black
                        # Clear queue if it's full
                        if self.frame_queue.full():
                            try:
                                self.frame_queue.get_nowait()
                            except queue.Empty:
                                pass
                        self.frame_queue.put(frame)
                    else:
                        # Black frame detected, try to adjust settings
                        self._adjust_camera_settings()
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        st.error("Camera stopped returning frames")
                        break
                        
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                break
    
    def _adjust_camera_settings(self):
        """Try to adjust camera settings to fix black frames"""
        try:
            # Try different exposure settings
            exposures = [25, 50, 75, -1]  # -1 for auto
            for exposure in exposures:
                self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None and np.mean(test_frame) > 10:
                    break
        except:
            pass
    
    def get_frame(self):
        """Get the latest frame from the queue"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def stop_camera(self):
        """Stop camera and clean up"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

def main():
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .violation-alert {
        background-color: #ffcccc;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff0000;
        margin: 0.5rem 0;
    }
    .camera-feed {
        border: 2px solid #1f77b4;
        border-radius: 10px;
        padding: 10px;
        background-color: #000000;
    }
    .black-frame-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # App header
    st.markdown('<h1 class="main-header">🛡️ PPE Safety Monitoring System</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'monitor' not in st.session_state:
        st.session_state.monitor = None
    if 'monitoring' not in st.session_state:
        st.session_state.monitoring = False
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
    if 'camera_manager' not in st.session_state:
        st.session_state.camera_manager = CameraManager()
    if 'camera_working' not in st.session_state:
        st.session_state.camera_working = False
    if 'use_test_image' not in st.session_state:
        st.session_state.use_test_image = False

    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose App Mode",
        ["🏠 Dashboard", "📹 Live Monitoring", "📊 Reports & Analytics", "⚙️ Camera Setup"]
    )
    
    # Dashboard
    if app_mode == "🏠 Dashboard":
        show_dashboard()
    
    # Live Monitoring
    elif app_mode == "📹 Live Monitoring":
        show_live_monitoring()
    
    # Reports & Analytics
    elif app_mode == "📊 Reports & Analytics":
        show_reports_analytics()
    
    # Camera Setup
    elif app_mode == "⚙️ Camera Setup":
        show_camera_setup()

def show_dashboard():
    st.header("📊 Safety Dashboard")
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        violations_count = 0 if st.session_state.monitor is None else len(st.session_state.monitor.violations_df)
        st.metric("Total Violations Today", violations_count)
    
    with col2:
        compliance_rate = 100  # Default
        if st.session_state.monitor and len(st.session_state.monitor.violations_df) > 0:
            total_detections = max(len(st.session_state.monitor.violations_df) * 10, 1)
            compliance_rate = 100 - (len(st.session_state.monitor.violations_df) / total_detections * 100)
        st.metric("Compliance Rate", f"{compliance_rate:.1f}%")
    
    with col3:
        most_violated = "No data"
        if st.session_state.monitor and not st.session_state.monitor.violations_df.empty:
            ppe_counts = {}
            for missing in st.session_state.monitor.violations_df['missing_ppe']:
                items = missing.split(', ')
                for item in items:
                    ppe_counts[item] = ppe_counts.get(item, 0) + 1
            if ppe_counts:
                most_violated = max(ppe_counts, key=ppe_counts.get)
        st.metric("Most Violated PPE", most_violated)
    
    with col4:
        current_shift = "Morning" if datetime.now().hour < 12 else "Afternoon" if datetime.now().hour < 18 else "Evening"
        st.metric("Current Shift", current_shift)
    
    st.markdown("---")
    
    # Charts section
    if st.session_state.monitor and not st.session_state.monitor.violations_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Violations by Hour")
            hourly_data = st.session_state.monitor.violations_df.groupby('shift_hour').size()
            fig = px.bar(
                x=hourly_data.index, 
                y=hourly_data.values,
                labels={'x': 'Hour of Day', 'y': 'Violations'},
                color=hourly_data.values,
                color_continuous_scale='reds'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Missing PPE Distribution")
            ppe_counts = {}
            for missing in st.session_state.monitor.violations_df['missing_ppe']:
                items = missing.split(', ')
                for item in items:
                    ppe_counts[item] = ppe_counts.get(item, 0) + 1
            
            if ppe_counts:
                fig = px.pie(
                    values=list(ppe_counts.values()),
                    names=list(ppe_counts.keys()),
                    title="Missing PPE Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No violation data available. Start monitoring to see charts.")
    
    # Recent violations
    st.subheader("🚨 Recent Violations")
    if st.session_state.monitor and not st.session_state.monitor.violations_df.empty:
        recent_violations = st.session_state.monitor.violations_df.tail(5)
        for _, violation in recent_violations.iterrows():
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"""
                    <div class="violation-alert">
                    <strong>Time:</strong> {violation['timestamp'].strftime('%H:%M:%S')} | 
                    <strong>Missing:</strong> {violation['missing_ppe']} | 
                    <strong>Confidence:</strong> {violation['confidence']:.2f}
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    if st.button("View Image", key=f"view_{violation.name}"):
                        try:
                            img = Image.open(violation['image_path'])
                            st.image(img, caption="Violation Evidence", use_column_width=True)
                        except:
                            st.error("Image not found")
    else:
        st.info("No recent violations detected.")

def show_live_monitoring():
    st.header("📹 Live PPE Monitoring")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Monitoring controls
        st.subheader("Monitoring Controls")
        
        # Initialize model
        if st.session_state.monitor is None:
            model_path = r"D:\runs\detect\train\weights\best.pt"
            if st.button("🚀 Initialize Monitoring System", type="primary"):
                try:
                    if os.path.exists(model_path):
                        st.session_state.monitor = PPEViolationMonitor(model_path)
                        st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                        st.success("Monitoring system initialized!")
                    else:
                        st.error(f"Model file not found at: {model_path}")
                except Exception as e:
                    st.error(f"Failed to initialize: {e}")
        
        # Start/Stop monitoring
        if st.session_state.monitor:
            if not st.session_state.monitoring:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("▶️ Start Camera Monitoring", type="primary"):
                        if st.session_state.camera_manager.start_camera():
                            st.session_state.monitoring = True
                            st.session_state.camera_working = True
                            st.rerun()
                        else:
                            st.error("Failed to start camera. Please check camera setup.")
                
                with col2:
                    if st.button("🖼️ Use Test Image Mode", type="secondary"):
                        st.session_state.use_test_image = True
                        st.session_state.monitoring = True
                        st.rerun()
            else:
                if st.button("⏹️ Stop Monitoring", type="secondary"):
                    st.session_state.monitoring = False
                    st.session_state.camera_working = False
                    st.session_state.use_test_image = False
                    st.session_state.camera_manager.stop_camera()
                    st.rerun()
    
    with col2:
        # Session info
        st.subheader("Session Information")
        if st.session_state.monitor:
            mode = "📷 Camera" if st.session_state.camera_working else "🖼️ Test Image" if st.session_state.use_test_image else "🔴 Inactive"
            st.info(f"""
            **Session ID:** {st.session_state.session_id}\n
            **Violations:** {len(st.session_state.monitor.violations_df)}\n
            **Mode:** {mode}\n
            **Status:** {'🟢 Active' if st.session_state.monitoring else '🔴 Inactive'}
            """)
    
    # Live feed section
    if st.session_state.monitoring and st.session_state.monitor:
        st.subheader("🔴 Live Feed")
        
        # Create placeholders
        video_placeholder = st.empty()
        status_placeholder = st.empty()
        violation_placeholder = st.empty()
        stats_placeholder = st.empty()
        warning_placeholder = st.empty()
        
        frame_count = 0
        last_violation_count = 0
        black_frame_count = 0
        
        try:
            while st.session_state.monitoring:
                frame = None
                
                if st.session_state.camera_working:
                    frame = st.session_state.camera_manager.get_frame()
                elif st.session_state.use_test_image:
                    # Use test image for demonstration
                    frame = generate_test_image()
                    time.sleep(0.5)  # Simulate camera delay
                
                if frame is not None:
                    frame_count += 1
                    
                    # Check for black frames
                    if np.mean(frame) < 10:  # Very dark frame
                        black_frame_count += 1
                        if black_frame_count > 10:
                            warning_placeholder.warning("""
                            **Black Frame Detected!** 
                            - Check camera connection
                            - Try different camera index
                            - Adjust camera settings
                            """)
                    
                    # Process frame
                    if frame_count % 3 == 0:  # Process every 3rd frame
                        annotated_frame, violations = st.session_state.monitor.process_frame(
                            frame, st.session_state.session_id
                        )
                        
                        # Convert frame for Streamlit
                        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        
                        # Display frame with black background container
                        with st.container():
                            st.markdown('<div class="camera-feed">', unsafe_allow_html=True)
                            video_placeholder.image(annotated_frame_rgb, channels="RGB", use_column_width=True, 
                                                  caption="Live Camera Feed with PPE Detection")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Update status
                        current_time = datetime.now().strftime('%H:%M:%S')
                        mode = "Camera" if st.session_state.camera_working else "Test Image"
                        status_placeholder.info(f"""
                        **Live Monitoring Active** | 
                        **Mode:** {mode} | 
                        **Violations:** {len(st.session_state.monitor.violations_df)} | 
                        **Last Check:** {current_time}
                        """)
                        
                        # Show recent violations
                        if violations:
                            violation_placeholder.warning(f"🚨 Violation Detected! Missing: {violations[0]['missing_ppe']}")
                        
                        # Update stats periodically
                        if frame_count % 30 == 0:
                            new_violations = len(st.session_state.monitor.violations_df) - last_violation_count
                            stats_placeholder.write(f"**Performance:** Processed {frame_count} frames | New violations: {new_violations}")
                            last_violation_count = len(st.session_state.monitor.violations_df)
                
                # Small delay to prevent high CPU usage
                time.sleep(0.1)
                
        except Exception as e:
            st.error(f"Monitoring error: {e}")
        finally:
            if st.session_state.monitoring:
                st.session_state.monitoring = False
                st.session_state.camera_manager.stop_camera()
    else:
        if st.session_state.monitor:
            st.info("Click 'Start Camera Monitoring' to begin live detection or 'Use Test Image Mode' for demonstration.")

def generate_test_image():
    """Generate a test image with simulated person and PPE"""
    # Create a simple test image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img.fill(200)  # Light gray background
    
    # Draw a simple "person"
    cv2.rectangle(img, (200, 100), (440, 400), (0, 255, 0), 2)
    cv2.putText(img, "Person", (250, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Randomly include/exclude PPE for testing
    import random
    if random.random() > 0.7:
        cv2.rectangle(img, (250, 50), (390, 90), (255, 0, 0), -1)  # Helmet
        cv2.putText(img, "Helmet", (260, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if random.random() > 0.7:
        cv2.rectangle(img, (220, 120), (420, 200), (0, 0, 255), -1)  # Vest
        cv2.putText(img, "Vest", (280, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img

def show_reports_analytics():
    st.header("📊 Reports & Analytics")
    
    if st.session_state.monitor is None:
        st.warning("Please initialize the monitoring system first from the Live Monitoring tab.")
        return
    
    # Report generation
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📈 Generate Excel Report", type="primary"):
            generate_excel_report()
    
    with col2:
        if st.button("📊 Generate Analytics Plots"):
            generate_analytics_plots()
    
    with col3:
        if st.button("🔄 Refresh Data"):
            st.rerun()
    
    st.markdown("---")
    
    # Data overview
    st.subheader("Data Overview")
    if not st.session_state.monitor.violations_df.empty:
        st.dataframe(
            st.session_state.monitor.violations_df[
                ['timestamp', 'missing_ppe', 'confidence', 'shift_hour']
            ].sort_values('timestamp', ascending=False),
            use_container_width=True
        )
        
        # Summary statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Summary Statistics")
            total_violations = len(st.session_state.monitor.violations_df)
            avg_confidence = st.session_state.monitor.violations_df['confidence'].mean()
            unique_persons = st.session_state.monitor.violations_df['person_id'].nunique()
            
            st.metric("Total Violations", total_violations)
            st.metric("Average Confidence", f"{avg_confidence:.3f}")
            st.metric("Unique Persons", unique_persons)
        
        with col2:
            st.subheader("Export Data")
            # CSV download
            csv = st.session_state.monitor.violations_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"ppe_violations_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    else:
        st.info("No violation data available for reporting.")

def show_camera_setup():
    st.header("⚙️ Camera Setup & Testing")
    
    st.subheader("Camera Test")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎥 Test Camera Access", type="primary"):
            test_camera_access()
    
    with col2:
        if st.button("🔄 Test Different Camera Indices"):
            test_multiple_cameras()
    
    st.subheader("Quick Fixes for Black Screen")
    
    fix_col1, fix_col2 = st.columns(2)
    
    with fix_col1:
        st.markdown("""
        **Immediate Solutions:**
        1. 🔄 **Restart Streamlit** - Close and reopen the app
        2. 📱 **Close other apps** - Zoom, Teams, browser camera access
        3. 🔌 **Reconnect camera** - Unplug and replug USB camera
        4. 🖥️ **Try different USB port** - Some ports work better
        """)
    
    with fix_col2:
        st.markdown("""
        **Advanced Solutions:**
        1. ⚙️ **Use DSHOW backend** - Added in this version
        2. 📹 **Try camera index 1, 2** - Not just index 0
        3. 🔧 **Update camera drivers** - Visit manufacturer website
        4. 🎮 **Use external webcam** - Often more reliable
        """)
    
    st.subheader("Alternative Camera Options")
    
    alt_col1, alt_col2 = st.columns(2)
    
    with alt_col1:
        if st.button("📱 Use Phone as Camera"):
            st.info("""
            **Using IP Webcam (Android):**
            1. Install 'IP Webcam' app
            2. Start server in app
            3. Note the IP address shown
            4. Use that URL in the app
            """)
    
    with alt_col2:
        if st.button("🎬 Use Video File"):
            uploaded_video = st.file_uploader("Upload video file", type=['mp4', 'avi', 'mov'])
            if uploaded_video:
                st.success("Video uploaded! Use Test Image mode for now.")

def test_camera_access():
    """Test camera access with different backends and settings"""
    st.info("Testing camera access with different configurations...")
    
    camera_indices = [0, 1, 2]
    backends = [
        (cv2.CAP_DSHOW, "DirectShow (Windows)"),
        (cv2.CAP_MSMF, "Microsoft Media Foundation"),
        (cv2.CAP_ANY, "Auto-Detect")
    ]
    
    for camera_index in camera_indices:
        st.write(f"### Testing Camera Index {camera_index}")
        
        for backend, backend_name in backends:
            cap = None
            try:
                cap = cv2.VideoCapture(camera_index, backend)
                
                if cap.isOpened():
                    # Try to read a frame
                    ret, frame = cap.read()
                    
                    if ret and frame is not None:
                        # Check if frame is not black
                        if np.mean(frame) > 10:
                            st.success(f"✅ **{backend_name}** - Working! Frame brightness: {np.mean(frame):.1f}")
                            
                            # Display the frame
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            st.image(frame_rgb, caption=f"Camera {camera_index} - {backend_name}", use_column_width=True)
                            
                            # Show frame info
                            st.write(f"Frame size: {frame.shape[1]}x{frame.shape[0]}")
                            break
                        else:
                            st.warning(f"⚠️ **{backend_name}** - Black frame detected")
                    else:
                        st.error(f"❌ **{backend_name}** - No frame received")
                else:
                    st.error(f"❌ **{backend_name}** - Cannot open camera")
                    
            except Exception as e:
                st.error(f"❌ **{backend_name}** - Error: {e}")
            finally:
                if cap:
                    cap.release()

def test_multiple_cameras():
    """Test multiple camera indices"""
    st.info("Scanning for available cameras...")
    
    available_cameras = []
    for i in range(5):  # Check first 5 indices
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use DSHOW for Windows
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                available_cameras.append(i)
                st.success(f"📷 Camera found at index {i}")
                cap.release()
            else:
                st.warning(f"❌ Camera at index {i} opened but no frame")
                cap.release()
        else:
            st.warning(f"❌ No camera at index {i}")
    
    if available_cameras:
        st.success(f"🎯 Available cameras: {available_cameras}")
        st.session_state.available_cameras = available_cameras
    else:
        st.error("❌ No cameras found!")

def generate_excel_report():
    """Generate Excel report"""
    if st.session_state.monitor.violations_df.empty:
        st.warning("No data available to generate report.")
        return
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = st.session_state.monitor.reports_dir / f"ppe_report_{timestamp}.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Main violations sheet
            st.session_state.monitor.violations_df.to_excel(
                writer, sheet_name='Violations_Details', index=False
            )
            
            # Summary sheet
            summary_data = {
                'Metric': ['Total Violations', 'Session Duration', 'Average Confidence'],
                'Value': [
                    len(st.session_state.monitor.violations_df),
                    f"{(datetime.now() - st.session_state.monitor.session_start).total_seconds() / 60:.1f} min",
                    f"{st.session_state.monitor.violations_df['confidence'].mean():.3f}"
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        st.success(f"✅ Excel report generated: {excel_path}")
        
        # Provide download link
        with open(excel_path, "rb") as f:
            excel_data = f.read()
        
        st.download_button(
            label="📥 Download Excel Report",
            data=excel_data,
            file_name=f"ppe_report_{timestamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    except Exception as e:
        st.error(f"Error generating report: {e}")

def generate_analytics_plots():
    """Generate analytical plots"""
    if st.session_state.monitor.violations_df.empty:
        st.warning("No data available for analytics.")
        return
    
    try:
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Violations by Hour', 'Missing PPE Distribution', 
                          'Confidence Distribution', 'Violation Trend'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        # Plot 1: Hourly violations
        hourly_data = st.session_state.monitor.violations_df.groupby('shift_hour').size()
        fig.add_trace(
            go.Bar(x=hourly_data.index, y=hourly_data.values, name='Violations'),
            row=1, col=1
        )
        
        # Plot 2: PPE distribution
        ppe_counts = {}
        for missing in st.session_state.monitor.violations_df['missing_ppe']:
            items = missing.split(', ')
            for item in items:
                ppe_counts[item] = ppe_counts.get(item, 0) + 1
        
        if ppe_counts:
            fig.add_trace(
                go.Pie(labels=list(ppe_counts.keys()), values=list(ppe_counts.values())),
                row=1, col=2
            )
        
        # Plot 3: Confidence distribution
        fig.add_trace(
            go.Histogram(x=st.session_state.monitor.violations_df['confidence'], name='Confidence'),
            row=2, col=1
        )
        
        # Plot 4: Time trend
        if len(st.session_state.monitor.violations_df) > 1:
            time_series = st.session_state.monitor.violations_df.set_index('timestamp').resample('10T').size()
            fig.add_trace(
                go.Scatter(x=time_series.index, y=time_series.values, mode='lines+markers', name='Trend'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=False, title_text="PPE Violation Analytics")
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error generating plots: {e}")

if __name__ == "__main__":
    main()
