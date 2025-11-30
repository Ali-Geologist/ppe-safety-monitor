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

def main():
    st.title("🛡️ PPE Safety Monitoring System")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Settings", "Live Monitoring", "Dashboard", "Reports"])
    
    if page == "Settings":
        show_settings()
    elif page == "Live Monitoring":
        show_live_monitoring()
    elif page == "Dashboard":
        show_dashboard()
    elif page == "Reports":
        show_reports()

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
    
    # Common PPE items - you can customize this mapping
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
            # Find matching class in model
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
    
    # Show available classes for reference
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
    
    # Map speed settings to actual parameters
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
    
    # Warning if no PPE selected
    if not selected_ppe:
        st.warning("⚠️ Please select at least one PPE item to monitor.")

def show_live_monitoring():
    st.header("📹 Live PPE Monitoring")
    
    # Check if settings are configured
    if not st.session_state.selected_ppe:
        st.warning("⚠️ Please configure detection settings first!")
        st.info("Go to the **Settings** page to select which PPE items to monitor.")
        return
    
    # Initialize model
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
            ["Webcam", "Test Mode", "Upload Video"],
            help="Webcam: Use your camera, Test Mode: Simulated detection, Upload: Use video file"
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
        if camera_mode == "Webcam":
            if st.button("🎥 Start Webcam", type="primary"):
                st.session_state.monitoring = True
                start_webcam_monitoring()
            
            if st.button("⏹️ Stop Monitoring"):
                st.session_state.monitoring = False
                st.rerun()
                
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
    """Optimized webcam monitoring with performance settings"""
    st.info("🚀 Starting optimized webcam monitoring...")
    
    # Get settings
    confidence = st.session_state.detection_settings['confidence']
    frame_skip = st.session_state.detection_settings['frame_skip']
    speed_params = st.session_state.detection_settings['speed_params']
    selected_classes = list(st.session_state.selected_ppe.keys())
    
    # Try different camera backends
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
    cap = None
    
    for backend in backends:
        try:
            cap = cv2.VideoCapture(0, backend)
            if cap.isOpened():
                break
        except:
            continue
    
    if cap is None or not cap.isOpened():
        st.error("❌ Cannot access webcam. Switching to Test Mode...")
        start_test_mode()
        return
    
    # Optimize camera settings for speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
    
    # Create placeholders
    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    performance_placeholder = st.empty()
    
    frame_count = 0
    processing_times = []
    last_fps_update = time.time()
    fps = 0
    
    while st.session_state.monitoring:
        try:
            start_time = time.time()
            ret, frame = cap.read()
            
            if not ret:
                st.error("Failed to read from camera")
                break
            
            frame_count += 1
            
            # Skip frames based on setting for better performance
            if frame_count % frame_skip == 0:
                # Run optimized detection
                results = st.session_state.model(
                    frame, 
                    conf=confidence,
                    classes=selected_classes,
                    verbose=False,
                    **speed_params
                )
                
                # Check for violations with selected PPE
                violations = check_for_violations(results, selected_classes)
                
                # Draw results
                annotated_frame = results[0].plot()
                
                # Add performance overlay
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Convert to RGB for Streamlit
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Display
                frame_placeholder.image(annotated_frame_rgb, caption="Live Camera Feed", use_column_width=True)
                
                # Update status
                if violations:
                    status_placeholder.warning(f"🚨 Missing: {', '.join(violations)}")
                    save_violation(frame, violations)
                else:
                    status_placeholder.success("✅ All required PPE detected")
            
            # Calculate FPS
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # Update FPS every second
            if time.time() - last_fps_update > 1.0:
                if processing_times:
                    avg_time = np.mean(processing_times)
                    fps = 1.0 / avg_time if avg_time > 0 else 0
                    processing_times = []
                last_fps_update = time.time()
                
                # Update performance stats
                performance_placeholder.info(
                    f"**Performance:** {fps:.1f} FPS | "
                    f"Frame skip: {frame_skip} | "
                    f"Processing: {avg_time*1000:.1f}ms"
                )
            
        except Exception as e:
            st.error(f"Camera error: {e}")
            break
    
    # Cleanup
    if cap:
        cap.release()

def start_test_mode():
    """Optimized test mode with customizable PPE simulation"""
    st.success("🎯 Test Mode Active - Custom PPE Detection Simulation")
    
    # Get settings
    selected_ppe = st.session_state.selected_ppe
    selected_classes = list(selected_ppe.keys())
    
    # Create placeholders
    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    info_placeholder = st.empty()
    
    frame_count = 0
    
    while st.session_state.monitoring:
        try:
            # Create test image based on selected PPE
            test_image = create_custom_test_image(frame_count, selected_ppe)
            
            # Run detection
            results = st.session_state.model(
                test_image, 
                conf=st.session_state.detection_settings['confidence'],
                classes=selected_classes,
                verbose=False
            )
            
            # Check for violations with selected PPE
            violations = check_for_violations(results, selected_classes)
            
            # Draw results
            annotated_frame = results[0].plot()
            
            # Add info overlay
            cv2.putText(annotated_frame, "TEST MODE", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Display
            frame_placeholder.image(annotated_frame_rgb, caption="Test Mode - Custom PPE Detection", use_column_width=True)
            
            # Update status
            if violations:
                status_placeholder.warning(f"🚨 Missing: {', '.join(violations)}")
                info_placeholder.info("Simulation: PPE violations detected")
                
                # Save sample violation occasionally
                if frame_count % 30 == 0:
                    save_violation(test_image, violations)
            else:
                status_placeholder.success("✅ All required PPE detected")
                info_placeholder.info("Simulation: All safety equipment present")
            
            frame_count += 1
            time.sleep(0.3)  # Controlled update rate
            
        except Exception as e:
            st.error(f"Test mode error: {e}")
            break

def create_custom_test_image(frame_count, selected_ppe):
    """Create test image based on selected PPE items"""
    # Create background
    img = np.ones((480, 640, 3), dtype=np.uint8) * 150
    
    # Draw person
    cv2.rectangle(img, (200, 100), (440, 400), (0, 255, 0), 2)
    cv2.putText(img, "Person", (250, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Get selected PPE names for display
    ppe_names = list(selected_ppe.values())
    
    # Simulate different scenarios
    scenario = (frame_count // 40) % (len(ppe_names) + 1)
    
    # Default: All PPE present
    missing_items = []
    
    if scenario > 0:
        # Missing one specific PPE item (rotate through selected items)
        missing_index = (scenario - 1) % len(ppe_names)
        missing_items = [ppe_names[missing_index]]
    
    # Draw PPE items that are present
    y_pos = 50
    for i, ppe_name in enumerate(ppe_names):
        if ppe_name not in missing_items:
            # Draw the PPE item
            color = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)][i % 5]
            cv2.rectangle(img, (250, y_pos), (390, y_pos + 40), color, -1)
            cv2.putText(img, ppe_name, (260, y_pos + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 50
    
    # Show missing items
    if missing_items:
        cv2.putText(img, f"MISSING: {', '.join(missing_items)}", 
                   (200, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return img

def process_uploaded_video(uploaded_file):
    """Process uploaded video file with custom settings"""
    # Get settings
    confidence = st.session_state.detection_settings['confidence']
    selected_classes = list(st.session_state.selected_ppe.keys())
    speed_params = st.session_state.detection_settings['speed_params']
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name
    
    # Process video
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
        
        # Process frame with custom settings
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
        
        # Display
        frame_placeholder.image(annotated_frame_rgb, caption="Video Processing", use_column_width=True)
        
        # Update status
        if violations:
            status_placeholder.warning(f"Violations: {', '.join(violations)}")
            save_violation(frame, violations)
        else:
            status_placeholder.info("No violations detected")
        
        # Update progress
        frame_count += 1
        progress_bar.progress(frame_count / total_frames)
        
        time.sleep(0.03)
    
    cap.release()
    os.unlink(video_path)
    st.success("Video processing completed!")

def check_for_violations(results, required_classes):
    """Check detection results for specific PPE violations"""
    detected_classes = set()
    
    if results and len(results) > 0:
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            detected_classes.add(class_id)
    
    # Only check for violations in selected PPE items
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
    
    # Convert violations to DataFrame
    df = pd.DataFrame([
        {
            'timestamp': v['timestamp'],
            'missing_ppe': v['missing_ppe'],
            'hour': v['timestamp'].hour,
            'selected_ppe': ', '.join(v['selected_ppe'])
        }
        for v in st.session_state.violations
    ])
    
    # Metrics
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
    
    # Charts
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
    
    # Recent violations with configuration context
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
    
    # Configuration context
    st.info(f"**Current Monitoring Configuration:** {', '.join(st.session_state.selected_ppe.values())}")
    
    # Generate report
    if st.button("📊 Generate Excel Report", type="primary"):
        generate_excel_report()
    
    # Data table
    st.subheader("Violation Data")
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
    
    # Export options
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
    """Generate comprehensive Excel report"""
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
    
    # Create Excel file in memory
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Violations sheet
        df.to_excel(writer, sheet_name='Violations', index=False)
        
        # Summary sheet
        summary_data = {
            'Total_Violations': [len(df)],
            'Date_Generated': [datetime.now()],
            'Monitoring_Configuration': [', '.join(st.session_state.selected_ppe.values())],
            'Most_Common_Violation': [df['Missing_PPE'].mode()[0] if not df.empty else 'None'],
            'Confidence_Setting': [st.session_state.detection_settings['confidence']],
            'Speed_Setting': [st.session_state.detection_settings['speed']]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        # Configuration sheet
        config_data = {
            'PPE_Item': list(st.session_state.selected_ppe.values()),
            'Class_ID': list(st.session_state.selected_ppe.keys())
        }
        pd.DataFrame(config_data).to_excel(writer, sheet_name='Configuration', index=False)
    
    # Download button
    st.download_button(
        "📥 Download Excel Report",
        output.getvalue(),
        f"ppe_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        "application/vnd.openformats-officedocument.spreadsheetml.sheet"
    )

if __name__ == "__main__":
    main()
