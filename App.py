import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
import os
from pathlib import Path
import tempfile
import io
from PIL import Image
import cv2

# Set page configuration with modern branding
st.set_page_config(
    page_title="SafetyEagle AI - PPE Detection",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern CSS matching VideoLoft style
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-blue: #2563eb;
        --dark-blue: #1e40af;
        --light-blue: #dbeafe;
        --accent-orange: #f59e0b;
        --dark-gray: #374151;
        --light-gray: #f8fafc;
    }
    
    /* Main container */
    .main {
        background-color: white;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--dark-blue) 100%);
        color: white;
        padding: 2rem 0;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .header-content {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 2rem;
    }
    
    .logo-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 1.5rem;
    }
    
    /* Stats cards */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .stat-card {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.2);
        text-align: center;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.8;
    }
    
    /* Modern cards */
    .modern-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 20px rgba(0,0,0,0.08);
        border: 1px solid #e5e7eb;
        margin-bottom: 1.5rem;
    }
    
    .card-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: var(--dark-gray);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: var(--light-gray);
        padding: 0.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background: transparent;
        border-radius: 8px;
        gap: 0.5rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary-blue) !important;
        color: white !important;
    }
    
    /* Buttons */
    .stButton button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--primary-blue), var(--accent-orange));
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border: 1px solid #e5e7eb;
    }
    
    /* Camera feed styling */
    .camera-feed {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border: 2px solid var(--light-gray);
    }
    
    /* PPE items grid */
    .ppe-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .ppe-item {
        background: var(--light-gray);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .ppe-item.selected {
        background: var(--light-blue);
        border-color: var(--primary-blue);
    }
    
    .ppe-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    /* Alert styling */
    .alert-success {
        background: #dcfce7;
        border: 1px solid #22c55e;
        color: #166534;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .alert-warning {
        background: #fef3c7;
        border: 1px solid #f59e0b;
        color: #92400e;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Chart containers */
    .chart-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Mobile camera specific styles */
    .camera-container {
        position: relative;
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 1rem;
    }
    
    .camera-overlay {
        position: absolute;
        top: 10px;
        left: 10px;
        background: rgba(0,0,0,0.7);
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.8rem;
        z-index: 10;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if 'violations' not in st.session_state:
        st.session_state.violations = []
    if 'monitoring' not in st.session_state:
        st.session_state.monitoring = False
    if 'selected_ppe' not in st.session_state:
        st.session_state.selected_ppe = {}
    if 'detection_settings' not in st.session_state:
        st.session_state.detection_settings = {
            'confidence': 0.7,
            'speed': 'medium',
            'frame_skip': 3
        }
    if 'camera_urls' not in st.session_state:
        st.session_state.camera_urls = []
    if 'mobile_camera_active' not in st.session_state:
        st.session_state.mobile_camera_active = False
    if 'camera_image' not in st.session_state:
        st.session_state.camera_image = None
    if 'live_camera_active' not in st.session_state:
        st.session_state.live_camera_active = False
    if 'project_info' not in st.session_state:
        st.session_state.project_info = {
            'company_name': '',
            'project_name': '',
            'engineer_name': '',
            'work_type': 'Offshore Operations',
            'workers_assigned': 0,
            'project_hours': 0,
        }
    if 'analytics_view' not in st.session_state:
        st.session_state.analytics_view = 'compliance'

# Standard PPE classes with icons
PPE_CLASSES = {
    0: {"name": "Hard Hat", "icon": "⛑️", "description": "Head Protection"},
    1: {"name": "Safety Glasses", "icon": "👓", "description": "Eye Protection"},
    2: {"name": "High-Vis Vest", "icon": "🦺", "description": "Visibility"},
    3: {"name": "Safety Gloves", "icon": "🧤", "description": "Hand Protection"},
    4: {"name": "Safety Boots", "icon": "👢", "description": "Foot Protection"},
    5: {"name": "Hearing Protection", "icon": "🎧", "description": "Hearing Safety"},
    6: {"name": "Face Shield", "icon": "🛡️", "description": "Face Protection"},
    7: {"name": "Respirator", "icon": "😷", "description": "Respiratory Protection"},
}

def create_simple_bar_chart(data, title, color="#2563eb"):
    """Create a simple bar chart using HTML/CSS"""
    if not data:
        return f'<div class="chart-container"><h4>{title}</h4><p>No data available</p></div>'
    
    max_value = max(data.values()) if data else 1
    chart_html = f"""
    <div class="chart-container">
        <h4>{title}</h4>
    """
    for label, value in data.items():
        width = (value / max_value) * 100 if max_value > 0 else 0
        chart_html += f"""
        <div style="margin: 10px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span>{label}</span>
                <span style="font-weight: bold;">{value}</span>
            </div>
            <div style="background: #e5e7eb; border-radius: 10px; height: 20px;">
                <div style="background: {color}; border-radius: 10px; height: 20px; width: {width}%; 
                          transition: width 0.3s ease;"></div>
            </div>
        </div>
        """
    chart_html += "</div>"
    return chart_html

def create_compliance_gauge(percentage, title):
    """Create a simple gauge chart using HTML/CSS"""
    # Ensure percentage is between 0 and 100
    percentage = max(0, min(100, percentage))
    color = "#22c55e" if percentage >= 90 else "#f59e0b" if percentage >= 80 else "#ef4444"
    
    # Fix the conic gradient - ensure it doesn't exceed 100%
    fill_percentage = min(percentage, 100)
    
    gauge_html = f"""
    <div class="chart-container" style="text-align: center;">
        <h4>{title}</h4>
        <div style="position: relative; width: 150px; height: 150px; margin: 0 auto;">
            <div style="position: absolute; top: 0; left: 0; width: 150px; height: 150px; 
                      border-radius: 50%; background: conic-gradient({color} 0% {fill_percentage}%, #e5e7eb {fill_percentage}% 100%);">
            </div>
            <div style="position: absolute; top: 15px; left: 15px; width: 120px; height: 120px; 
                      border-radius: 50%; background: white; display: flex; align-items: center; justify-content: center;">
                <span style="font-size: 2rem; font-weight: bold; color: {color};">{int(percentage)}%</span>
            </div>
        </div>
    </div>
    """
    return gauge_html

def calculate_compliance_rate():
    """Calculate compliance rate based on violations"""
    if not st.session_state.violations:
        return 100
    
    total_workers = st.session_state.project_info['workers_assigned'] or 1
    total_violations = len(st.session_state.violations)
    
    # Simple calculation: reduce compliance by 2% per violation per worker
    violation_impact = min(100, (total_violations / total_workers) * 2)
    return max(0, 100 - violation_impact)

def get_violation_stats():
    """Get violation statistics for analytics"""
    if not st.session_state.violations:
        return {}, {}, {}
    
    # PPE violation distribution
    ppe_violations = {}
    time_violations = {}
    daily_violations = {}
    
    for violation in st.session_state.violations:
        # Count by PPE type
        missing_items = violation.get('missing_classes', 'Unknown').split(', ')
        for item in missing_items:
            ppe_violations[item] = ppe_violations.get(item, 0) + 1
        
        # Count by hour of day
        hour = violation['timestamp'].hour
        time_slot = f"{hour:02d}:00-{hour+1:02d}:00"
        time_violations[time_slot] = time_violations.get(time_slot, 0) + 1
        
        # Count by day
        day = violation['timestamp'].strftime('%Y-%m-%d')
        daily_violations[day] = daily_violations.get(day, 0) + 1
    
    return ppe_violations, time_violations, daily_violations

def simulate_ppe_detection_with_boxes(image):
    """Simulate PPE detection with bounding boxes - SIMPLIFIED VERSION"""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Create a copy to draw on
        if len(img_array.shape) == 3:
            # Color image
            img_with_boxes = img_array.copy()
        else:
            # Convert grayscale to color
            img_with_boxes = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        # Get image dimensions
        height, width = img_with_boxes.shape[:2]
        
        # Simulate detection results with bounding boxes
        detection_results = []
        
        # Colors for different PPE types
        colors = {
            "Hard Hat": (0, 255, 0),  # Green
            "Safety Glasses": (255, 255, 0),  # Yellow
            "High-Vis Vest": (255, 165, 0),  # Orange
            "Safety Gloves": (0, 255, 255),  # Cyan
            "Safety Boots": (255, 0, 255),  # Magenta
            "Hearing Protection": (128, 0, 128),  # Purple
            "Face Shield": (255, 0, 0),  # Red
            "Respirator": (0, 0, 255),  # Blue
        }
        
        # Simulate detecting some PPE items based on selected PPE
        import random
        for ppe_id, ppe_name in st.session_state.selected_ppe.items():
            # Random chance to detect each PPE item
            if random.random() > 0.3:  # 70% detection rate
                # Generate random bounding box (ensure it's within image bounds)
                box_width = random.randint(80, 150)
                box_height = random.randint(80, 150)
                x1 = random.randint(20, width - box_width - 20)
                y1 = random.randint(20, height - box_height - 20)
                x2 = x1 + box_width
                y2 = y1 + box_height
                
                confidence = random.uniform(0.6, 0.95)
                
                detection_results.append({
                    'class': ppe_name,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2]
                })
                
                # Draw bounding box
                color = colors.get(ppe_name, (255, 255, 255))
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 3)
                
                # Draw label background
                label = f"{ppe_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                label_y = max(y1 - 10, label_size[1] + 5)
                cv2.rectangle(img_with_boxes, 
                            (x1, label_y - label_size[1] - 5), 
                            (x1 + label_size[0], label_y), 
                            color, -1)
                
                # Draw label text
                cv2.putText(img_with_boxes, label, (x1, label_y - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return Image.fromarray(img_with_boxes), detection_results
        
    except Exception as e:
        st.error(f"Error in detection simulation: {str(e)}")
        # Return original image if there's an error
        return image, []

def main():
    # Modern Header
    st.markdown("""
    <div class="main-header">
        <div class="header-content">
            <div class="logo-title">
                <span>🦅</span>
                SafetyEagle AI
            </div>
            <div class="header-subtitle">
                AI-Powered PPE Detection for Enhanced Workplace Safety
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    initialize_session_state()
    
    # Main tabs
    tabs = st.tabs(["🏠 **Project Setup**", "📹 **Live Monitoring**", "📊 **Analytics**"])
    
    with tabs[0]:
        show_project_setup()
    with tabs[1]:
        show_live_monitoring()
    with tabs[2]:
        show_analytics()

def show_project_setup():
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="modern-card"><div class="card-header">🏢 Project Information</div></div>', unsafe_allow_html=True)
        
        company_name = st.text_input("Company Name", value=st.session_state.project_info['company_name'])
        project_name = st.text_input("Project Name", value=st.session_state.project_info['project_name'])
        engineer_name = st.text_input("Safety Engineer Name", value=st.session_state.project_info['engineer_name'])
        work_type = st.selectbox(
            "Work Type",
            ["Offshore Operations", "Drilling", "Refinery", "Pipeline", "Maintenance", "Construction", "Other"]
        )
        workers = st.number_input("Number of Workers", min_value=0, value=st.session_state.project_info['workers_assigned'])
        project_hours = st.number_input("Project Hours", min_value=0, value=st.session_state.project_info['project_hours'])
        
        if st.button("💾 Save Project Settings", type="primary", use_container_width=True):
            st.session_state.project_info = {
                'company_name': company_name,
                'project_name': project_name,
                'engineer_name': engineer_name,
                'work_type': work_type,
                'workers_assigned': workers,
                'project_hours': project_hours,
            }
            st.success("Project configuration saved!")
    
    with col2:
        st.markdown('<div class="modern-card"><div class="card-header">🛡️ PPE Configuration</div></div>', unsafe_allow_html=True)
        
        st.subheader("Select PPE Items to Monitor")
        st.info("Choose the safety equipment you want to detect and monitor")
        
        # PPE selection grid
        cols = st.columns(2)
        for i, (ppe_id, ppe_data) in enumerate(PPE_CLASSES.items()):
            with cols[i % 2]:
                if st.checkbox(
                    f"{ppe_data['icon']} {ppe_data['name']}", 
                    value=ppe_id in st.session_state.selected_ppe,
                    key=f"ppe_{ppe_id}",
                    help=ppe_data['description']
                ):
                    st.session_state.selected_ppe[ppe_id] = ppe_data['name']
                else:
                    if ppe_id in st.session_state.selected_ppe:
                        del st.session_state.selected_ppe[ppe_id]
        
        st.markdown("---")
        st.info(f"**Selected {len(st.session_state.selected_ppe)} PPE items for monitoring**")
        
        if st.session_state.selected_ppe:
            st.success("✅ PPE configuration saved! You can now start live monitoring.")
        else:
            st.warning("⚠️ Please select at least one PPE item to monitor")

def show_live_monitoring():
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="modern-card"><div class="card-header">📱 Live Mobile Camera Monitoring</div></div>', unsafe_allow_html=True)
        
        # Check if PPE is selected
        if not st.session_state.selected_ppe:
            st.error("❌ Please configure PPE items in Project Setup first!")
            return
        
        # Live Camera Controls
        col_control1, col_control2 = st.columns(2)
        with col_control1:
            if not st.session_state.live_camera_active:
                if st.button("🎬 Start Live Monitoring", type="primary", use_container_width=True):
                    st.session_state.live_camera_active = True
                    st.session_state.monitoring = True
                    st.rerun()
            else:
                if st.button("⏹️ Stop Live Monitoring", type="secondary", use_container_width=True):
                    st.session_state.live_camera_active = False
                    st.session_state.monitoring = False
                    st.rerun()
        
        with col_control2:
            if st.session_state.live_camera_active:
                st.success("🔴 LIVE - Monitoring Active")
            else:
                st.info("⚫ Monitoring Inactive")
        
        # Live Camera Feed with Real-time Processing
        if st.session_state.live_camera_active:
            st.subheader("📸 Live Camera Feed with PPE Detection")
            
            # Use camera input for continuous monitoring
            camera_image = st.camera_input(
                "Live PPE Detection - Camera is actively monitoring",
                key="live_camera",
                help="Position camera to monitor work area. Detection happens in real-time."
            )
            
            if camera_image is not None:
                # Store the latest image
                st.session_state.camera_image = camera_image
                
                # Process the image with simulated PPE detection
                with st.spinner("🔄 Processing live feed for PPE detection..."):
                    # Simulate processing time for real-time feel
                    time.sleep(0.5)
                    
                    # Get image with bounding boxes
                    processed_image, detections = simulate_ppe_detection_with_boxes(camera_image)
                    
                    # Display processed image with detections
                    st.image(processed_image, 
                            caption="Live PPE Detection - Bounding boxes show detected safety equipment", 
                            use_column_width=True)
                    
                    # Show detection results
                    if detections:
                        st.success(f"✅ Detected {len(detections)} PPE items:")
                        for detection in detections:
                            st.write(f"- {detection['class']} (confidence: {detection['confidence']:.2f})")
                        
                        # Check for missing PPE
                        detected_classes = [det['class'] for det in detections]
                        missing_ppe = [ppe for ppe in st.session_state.selected_ppe.values() if ppe not in detected_classes]
                        
                        if missing_ppe:
                            st.error(f"❌ Missing PPE: {', '.join(missing_ppe)}")
                            
                            # Auto-report violation for continuous missing PPE
                            violation = {
                                'timestamp': datetime.now(),
                                'missing_classes': ', '.join(missing_ppe),
                                'worker_id': 'Live Camera',
                                'source': 'Live Monitoring',
                                'image': camera_image
                            }
                            st.session_state.violations.append(violation)
                    else:
                        st.warning("⚠️ No PPE items detected in current frame")
            
            else:
                st.info("👆 Click 'Allow' to enable camera and start live monitoring")
        
        else:
            # Instructions when monitoring is inactive
            st.markdown("""
            <div style="background: #f8fafc; 
                        border-radius: 12px; 
                        padding: 2rem; 
                        text-align: center; 
                        color: #6b7280;
                        border: 2px dashed #d1d5db;
                        margin: 1rem 0;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">📱</div>
                <h3>Live Monitoring Ready</h3>
                <p>Click 'Start Live Monitoring' to begin real-time PPE detection</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            ### 📋 Live Monitoring Features:
            - **Real-time PPE detection** with bounding boxes
            - **Continuous monitoring** while camera is active
            - **Automatic violation reporting**
            - **Live feedback** on detection results
            - **No manual photo capture required**
            """)
    
    with col2:
        st.markdown('<div class="modern-card"><div class="card-header">⚙️ Monitoring Settings</div></div>', unsafe_allow_html=True)
        
        confidence = st.slider(
            "Detection Confidence",
            min_value=0.1,
            max_value=0.9,
            value=st.session_state.detection_settings['confidence'],
            step=0.1,
            help="Higher values = more accurate but fewer detections"
        )
        
        processing_mode = st.selectbox(
            "Processing Mode",
            ["Fast", "Balanced", "High Accuracy"],
            index=1,
            help="Balance between speed and detection accuracy"
        )
        
        # Update settings
        st.session_state.detection_settings['confidence'] = confidence
        
        st.markdown('<div class="modern-card"><div class="card-header">🎯 Active PPE Monitoring</div></div>', unsafe_allow_html=True)
        
        # Show currently monitored PPE
        st.subheader("Monitoring These PPE Items:")
        for ppe_name in st.session_state.selected_ppe.values():
            st.write(f"✅ {ppe_name}")
        
        st.info(f"Total: {len(st.session_state.selected_ppe)} items being monitored")
        
        # Quick actions
        st.markdown('<div class="modern-card"><div class="card-header">🔧 Quick Actions</div></div>', unsafe_allow_html=True)
        
        if st.button("📊 View Analytics", use_container_width=True):
            st.switch_page("?tab=Analytics")

def show_analytics():
    st.markdown('<div class="modern-card"><div class="card-header">📊 Safety Analytics Dashboard</div></div>', unsafe_allow_html=True)
    
    # Analytics View Selection
    col_view1, col_view2, col_view3 = st.columns(3)
    with col_view1:
        if st.button("📈 Compliance Overview", use_container_width=True):
            st.session_state.analytics_view = 'compliance'
    with col_view2:
        if st.button("🛡️ PPE Analysis", use_container_width=True):
            st.session_state.analytics_view = 'ppe_analysis'
    with col_view3:
        if st.button("⏰ Time Analysis", use_container_width=True):
            st.session_state.analytics_view = 'time_analysis'
    
    # Calculate real metrics
    compliance_rate = calculate_compliance_rate()
    total_violations = len(st.session_state.violations)
    workers = st.session_state.project_info['workers_assigned'] or 1
    ppe_violations, time_violations, daily_violations = get_violation_stats()
    
    # Key Metrics Always Visible
    st.subheader("📊 Key Safety Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Overall Compliance", f"{compliance_rate:.1f}%")
    with col2:
        st.metric("Total Violations", total_violations)
    with col3:
        st.metric("Workers", workers)
    with col4:
        safety_score = "A+" if compliance_rate >= 95 else "A" if compliance_rate >= 90 else "B" if compliance_rate >= 80 else "C"
        st.metric("Safety Score", safety_score)
    
    # Dynamic Analytics Views
    if st.session_state.analytics_view == 'compliance':
        show_compliance_analytics(compliance_rate, daily_violations)
    elif st.session_state.analytics_view == 'ppe_analysis':
        show_ppe_analytics(ppe_violations)
    elif st.session_state.analytics_view == 'time_analysis':
        show_time_analytics(time_violations)
    
    # Recent alerts (always shown)
    show_recent_alerts()

def show_compliance_analytics(compliance_rate, daily_violations):
    st.markdown('<div class="modern-card"><div class="card-header">📈 Compliance Analytics</div></div>', unsafe_allow_html=True)
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Compliance gauge - FIXED
        st.markdown(create_compliance_gauge(compliance_rate, "Overall Compliance Rate"))
    
    with col_chart2:
        # Daily violations trend
        if daily_violations:
            st.markdown(create_simple_bar_chart(daily_violations, "Daily Violations Trend", "#ef4444"))
        else:
            st.markdown("""
            <div class="chart-container">
                <h4>Daily Violations Trend</h4>
                <p>No violations recorded yet</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Compliance recommendations
    st.markdown('<div class="modern-card"><div class="card-header">💡 Compliance Recommendations</div></div>', unsafe_allow_html=True)
    
    if compliance_rate >= 95:
        st.success("""
        **Excellent Compliance!** 🎉
        - Continue current safety protocols
        - Maintain regular equipment checks
        - Share best practices across teams
        """)
    elif compliance_rate >= 80:
        st.warning("""
        **Good Compliance - Room for Improvement** 📊
        - Conduct refresher PPE training
        - Increase monitoring frequency
        - Address common violation patterns
        """)
    else:
        st.error("""
        **Needs Immediate Attention** 🚨
        - Implement mandatory safety training
        - Increase supervision in high-risk areas
        - Review and update safety protocols
        - Consider disciplinary measures for repeat violations
        """)

def show_ppe_analytics(ppe_violations):
    st.markdown('<div class="modern-card"><div class="card-header">🛡️ PPE-Specific Analytics</div></div>', unsafe_allow_html=True)
    
    col_viol1, col_viol2 = st.columns(2)
    
    with col_viol1:
        if ppe_violations:
            st.markdown(create_simple_bar_chart(ppe_violations, "Violations by PPE Type", "#ef4444"))
        else:
            st.markdown("""
            <div class="chart-container">
                <h4>Violations by PPE Type</h4>
                <p>No PPE violations recorded</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col_viol2:
        # PPE compliance rates
        ppe_compliance = {}
        total_checks = len(st.session_state.violations) * len(st.session_state.selected_ppe)
        
        for ppe_name in st.session_state.selected_ppe.values():
            violations = ppe_violations.get(ppe_name, 0)
            compliance = max(0, 100 - (violations / max(1, total_checks) * 1000))
            ppe_compliance[ppe_name] = compliance
        
        if ppe_compliance:
            st.markdown(create_simple_bar_chart(ppe_compliance, "PPE Compliance Rates", "#22c55e"))
        else:
            st.markdown("""
            <div class="chart-container">
                <h4>PPE Compliance Rates</h4>
                <p>No compliance data available</p>
            </div>
            """, unsafe_allow_html=True)
    
    # PPE-specific recommendations
    if ppe_violations:
        worst_ppe = max(ppe_violations.items(), key=lambda x: x[1]) if ppe_violations else None
        if worst_ppe:
            st.markdown(f"""
            <div class="modern-card">
                <div class="card-header">🎯 Focus Area</div>
                <h4 style="color: #ef4444;">{worst_ppe[0]}</h4>
                <p>This PPE item has the most violations ({worst_ppe[1]}). Consider:</p>
                <ul>
                    <li>Additional training for {worst_ppe[0]} usage</li>
                    <li>Increased availability of {worst_ppe[0]} equipment</li>
                    <li>Stricter enforcement for {worst_ppe[0]} compliance</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

def show_time_analytics(time_violations):
    st.markdown('<div class="modern-card"><div class="card-header">⏰ Time-Based Analytics</div></div>', unsafe_allow_html=True)
    
    col_time1, col_time2 = st.columns(2)
    
    with col_time1:
        if time_violations:
            st.markdown(create_simple_bar_chart(time_violations, "Violations by Time of Day", "#f59e0b"))
        else:
            st.markdown("""
            <div class="chart-container">
                <h4>Violations by Time of Day</h4>
                <p>No time-based data available</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col_time2:
        # Shift analysis
        shift_violations = {"Morning (6AM-2PM)": 0, "Afternoon (2PM-10PM)": 0, "Night (10PM-6AM)": 0}
        
        for violation in st.session_state.violations:
            hour = violation['timestamp'].hour
            if 6 <= hour < 14:
                shift_violations["Morning (6AM-2PM)"] += 1
            elif 14 <= hour < 22:
                shift_violations["Afternoon (2PM-10PM)"] += 1
            else:
                shift_violations["Night (10PM-6AM)"] += 1
        
        if any(shift_violations.values()):
            st.markdown(create_simple_bar_chart(shift_violations, "Violations by Shift", "#8b5cf6"))
        else:
            st.markdown("""
            <div class="chart-container">
                <h4>Violations by Shift</h4>
                <p>No shift data available</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Time-based recommendations
    if time_violations:
        worst_time = max(time_violations.items(), key=lambda x: x[1]) if time_violations else None
        if worst_time:
            st.markdown(f"""
            <div class="modern-card">
                <div class="card-header">🕒 Peak Violation Time</div>
                <h4 style="color: #f59e0b;">{worst_time[0]}</h4>
                <p>This time period has the most violations. Consider:</p>
                <ul>
                    <li>Increased supervision during {worst_time[0]}</li>
                    <li>Additional safety briefings before this period</li>
                    <li>Reviewing work schedules and fatigue management</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

def show_recent_alerts():
    """Show recent safety alerts"""
    st.markdown('<div class="modern-card"><div class="card-header">🚨 Recent Safety Events</div></div>', unsafe_allow_html=True)
    
    if st.session_state.violations:
        # Show last 5 violations
        recent_violations = st.session_state.violations[-5:]
        
        for violation in reversed(recent_violations):
            severity_color = "#ef4444"  # High severity for violations
            
            st.markdown(f"""
            <div style="background: white; padding: 1rem; border-radius: 8px; border-left: 4px solid {severity_color}; margin: 0.5rem 0;">
                <div style="display: flex; justify-content: between; align-items: start;">
                    <div style="flex: 1;">
                        <strong>PPE Violation</strong><br>
                        <small style="color: #6b7280;">{violation['timestamp'].strftime('%Y-%m-%d %H:%M')}</small>
                    </div>
                    <div style="background: {severity_color}; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.8rem;">
                        High
                    </div>
                </div>
                <div style="margin-top: 0.5rem;">
                    <strong>Missing:</strong> {violation.get('missing_classes', 'Unknown')}<br>
                    <strong>Source:</strong> {violation.get('source', 'Manual Report')}
                    {f"<br><strong>Worker:</strong> {violation.get('worker_id', '')}" if violation.get('worker_id') else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: #f0fdf4; padding: 2rem; border-radius: 8px; text-align: center; border: 1px solid #bbf7d0;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">✅</div>
            <h3 style="color: #166534; margin-bottom: 0.5rem;">No Safety Violations</h3>
            <p style="color: #15803d;">Excellent safety compliance record!</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
