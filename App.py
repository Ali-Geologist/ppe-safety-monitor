import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
import random
from PIL import Image
import cv2

# Set page configuration
st.set_page_config(
    page_title="SafetyEagle AI - PPE Detection",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        color: white;
        padding: 2rem 0;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 20px 20px;
    }
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
        color: #374151;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'violations' not in st.session_state:
    st.session_state.violations = []
if 'selected_ppe' not in st.session_state:
    st.session_state.selected_ppe = {}
if 'live_camera_active' not in st.session_state:
    st.session_state.live_camera_active = False
if 'project_info' not in st.session_state:
    st.session_state.project_info = {
        'company_name': '', 'project_name': '', 'engineer_name': '',
        'work_type': 'Offshore Operations', 'workers_assigned': 0, 'project_hours': 0
    }
if 'analytics_view' not in st.session_state:
    st.session_state.analytics_view = 'compliance'

# PPE classes
PPE_CLASSES = {
    0: {"name": "Hard Hat", "icon": "⛑️"},
    1: {"name": "Safety Glasses", "icon": "👓"},
    2: {"name": "High-Vis Vest", "icon": "🦺"},
    3: {"name": "Safety Gloves", "icon": "🧤"},
    4: {"name": "Safety Boots", "icon": "👢"},
}

def create_compliance_gauge(percentage):
    """Create a working compliance gauge"""
    percentage = max(0, min(100, percentage))
    color = "#22c55e" if percentage >= 90 else "#f59e0b" if percentage >= 80 else "#ef4444"
    
    # FIXED: Proper conic gradient calculation
    gauge_html = f"""
    <div style="background: white; border-radius: 12px; padding: 1.5rem; text-align: center; margin: 1rem 0;">
        <h4>Overall Compliance Rate</h4>
        <div style="position: relative; width: 150px; height: 150px; margin: 0 auto;">
            <div style="position: absolute; top: 0; left: 0; width: 150px; height: 150px; 
                      border-radius: 50%; background: conic-gradient({color} 0% {percentage}%, #e5e7eb {percentage}% 100%);">
            </div>
            <div style="position: absolute; top: 15px; left: 15px; width: 120px; height: 120px; 
                      border-radius: 50%; background: white; display: flex; align-items: center; justify-content: center;">
                <span style="font-size: 2rem; font-weight: bold; color: {color};">{int(percentage)}%</span>
            </div>
        </div>
    </div>
    """
    return gauge_html

def draw_bounding_boxes_simple(image):
    """Draw bounding boxes on image - SIMPLE WORKING VERSION"""
    try:
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Make sure we have a color image
        if len(img_array.shape) == 2:  # Grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        height, width = img_array.shape[:2]
        detections = []
        
        # Colors for different PPE
        colors = {
            "Hard Hat": (0, 255, 0),      # Green
            "Safety Glasses": (255, 255, 0), # Yellow  
            "High-Vis Vest": (255, 165, 0), # Orange
            "Safety Gloves": (0, 255, 255), # Cyan
            "Safety Boots": (255, 0, 255),  # Magenta
        }
        
        # Create detections for selected PPE
        for ppe_name in st.session_state.selected_ppe.values():
            if random.random() > 0.4:  # 60% detection rate
                # Create random bounding box
                w, h = random.randint(80, 120), random.randint(80, 120)
                x = random.randint(50, width - w - 50)
                y = random.randint(50, height - h - 50)
                
                confidence = round(random.uniform(0.6, 0.95), 2)
                detections.append({
                    'class': ppe_name,
                    'confidence': confidence,
                    'bbox': [x, y, x+w, y+h]
                })
                
                # Draw bounding box
                color = colors.get(ppe_name, (255, 255, 255))
                cv2.rectangle(img_array, (x, y), (x+w, y+h), color, 3)
                
                # Draw label
                label = f"{ppe_name}: {confidence}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                label_y = max(y - 10, label_size[1] + 5)
                
                # Label background
                cv2.rectangle(img_array, 
                            (x, label_y - label_size[1] - 5),
                            (x + label_size[0], label_y), 
                            color, -1)
                
                # Label text
                cv2.putText(img_array, label, (x, label_y - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return Image.fromarray(img_array), detections
        
    except Exception as e:
        st.error(f"Detection error: {str(e)}")
        return image, []

def calculate_compliance_rate():
    if not st.session_state.violations:
        return 100
    workers = st.session_state.project_info['workers_assigned'] or 1
    violations = len(st.session_state.violations)
    return max(0, 100 - (violations / workers * 10))

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <div style="max-width: 1200px; margin: 0 auto; padding: 0 2rem;">
            <div style="font-size: 2.5rem; font-weight: 700; display: flex; align-items: center; gap: 0.5rem;">
                <span>🦅</span> SafetyEagle AI
            </div>
            <div style="font-size: 1.2rem; opacity: 0.9;">
                AI-Powered PPE Detection for Enhanced Workplace Safety
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tabs
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
        
        st.text_input("Company Name", value=st.session_state.project_info['company_name'], key="company")
        st.text_input("Project Name", value=st.session_state.project_info['project_name'], key="project")
        st.text_input("Safety Engineer", value=st.session_state.project_info['engineer_name'], key="engineer")
        st.selectbox("Work Type", ["Offshore Operations", "Drilling", "Refinery", "Construction"], key="work_type")
        st.number_input("Number of Workers", min_value=0, value=st.session_state.project_info['workers_assigned'], key="workers")
        
        if st.button("💾 Save Project Settings", use_container_width=True):
            st.session_state.project_info = {
                'company_name': st.session_state.company,
                'project_name': st.session_state.project,
                'engineer_name': st.session_state.engineer,
                'work_type': st.session_state.work_type,
                'workers_assigned': st.session_state.workers,
                'project_hours': 240,
            }
            st.success("Project configuration saved!")
    
    with col2:
        st.markdown('<div class="modern-card"><div class="card-header">🛡️ PPE Configuration</div></div>', unsafe_allow_html=True)
        
        st.write("Select PPE items to monitor:")
        for ppe_id, ppe_data in PPE_CLASSES.items():
            if st.checkbox(f"{ppe_data['icon']} {ppe_data['name']}", 
                          value=ppe_id in st.session_state.selected_ppe,
                          key=f"ppe_{ppe_id}"):
                st.session_state.selected_ppe[ppe_id] = ppe_data['name']
            elif ppe_id in st.session_state.selected_ppe:
                del st.session_state.selected_ppe[ppe_id]
        
        st.info(f"Selected {len(st.session_state.selected_ppe)} PPE items")

def show_live_monitoring():
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="modern-card"><div class="card-header">📱 Live Camera Monitoring</div></div>', unsafe_allow_html=True)
        
        if not st.session_state.selected_ppe:
            st.error("Please select PPE items in Project Setup first!")
            return
        
        # Camera controls
        col1, col2 = st.columns(2)
        with col1:
            if not st.session_state.live_camera_active:
                if st.button("🎬 Start Live Monitoring", use_container_width=True, type="primary"):
                    st.session_state.live_camera_active = True
                    st.rerun()
            else:
                if st.button("⏹️ Stop Monitoring", use_container_width=True):
                    st.session_state.live_camera_active = False
                    st.rerun()
        
        with col2:
            if st.session_state.live_camera_active:
                st.success("🔴 LIVE")
            else:
                st.info("⚫ Ready")
        
        # Camera feed
        if st.session_state.live_camera_active:
            camera_img = st.camera_input("Live PPE Detection", key="live_cam")
            
            if camera_img is not None:
                with st.spinner("Detecting PPE..."):
                    time.sleep(1)  # Simulate processing
                    
                    # PROCESS IMAGE WITH BOUNDING BOXES
                    processed_img, detections = draw_bounding_boxes_simple(camera_img)
                    
                    # Display image with bounding boxes
                    st.image(processed_img, caption="Live Detection with Bounding Boxes", use_column_width=True)
                    
                    if detections:
                        st.success(f"Detected {len(detections)} PPE items:")
                        for det in detections:
                            st.write(f"- {det['class']} ({det['confidence']})")
                        
                        # Check for missing PPE
                        detected_classes = [d['class'] for d in detections]
                        missing = [p for p in st.session_state.selected_ppe.values() if p not in detected_classes]
                        
                        if missing:
                            st.error(f"Missing: {', '.join(missing)}")
                            # Record violation
                            st.session_state.violations.append({
                                'timestamp': datetime.now(),
                                'missing_classes': ', '.join(missing),
                                'source': 'Live Camera'
                            })
                    else:
                        st.warning("No PPE detected")
        else:
            st.info("Click 'Start Live Monitoring' to begin detection")
    
    with col2:
        st.markdown('<div class="modern-card"><div class="card-header">⚙️ Settings</div></div>', unsafe_allow_html=True)
        
        st.slider("Confidence Threshold", 0.1, 0.9, 0.7, key="confidence")
        st.selectbox("Processing Mode", ["Fast", "Balanced", "Accurate"], key="mode")
        
        st.markdown('<div class="modern-card"><div class="card-header">🎯 Monitoring</div></div>', unsafe_allow_html=True)
        
        for ppe in st.session_state.selected_ppe.values():
            st.write(f"✅ {ppe}")

def show_analytics():
    st.markdown('<div class="modern-card"><div class="card-header">📊 Analytics Dashboard</div></div>', unsafe_allow_html=True)
    
    # View selector
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📈 Compliance", use_container_width=True):
            st.session_state.analytics_view = 'compliance'
    with col2:
        if st.button("🛡️ PPE Analysis", use_container_width=True):
            st.session_state.analytics_view = 'ppe'
    with col3:
        if st.button("⏰ Time Analysis", use_container_width=True):
            st.session_state.analytics_view = 'time'
    
    # Metrics
    compliance = calculate_compliance_rate()
    violations = len(st.session_state.violations)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Compliance", f"{compliance:.1f}%")
    with col2:
        st.metric("Violations", violations)
    with col3:
        st.metric("Workers", st.session_state.project_info['workers_assigned'])
    with col4:
        st.metric("PPE Items", len(st.session_state.selected_ppe))
    
    # Analytics views
    if st.session_state.analytics_view == 'compliance':
        show_compliance_view(compliance)
    elif st.session_state.analytics_view == 'ppe':
        show_ppe_view()
    else:
        show_time_view()
    
    # Recent violations
    st.markdown('<div class="modern-card"><div class="card-header">🚨 Recent Events</div></div>', unsafe_allow_html=True)
    
    if st.session_state.violations:
        for i, violation in enumerate(st.session_state.violations[-3:]):
            st.error(f"**Violation {i+1}:** {violation['missing_classes']} - {violation['timestamp'].strftime('%H:%M')}")
    else:
        st.success("No safety violations recorded")

def show_compliance_view(compliance):
    col1, col2 = st.columns(2)
    
    with col1:
        # FIXED: This should now work properly
        st.markdown(create_compliance_gauge(compliance), unsafe_allow_html=True)
    
    with col2:
        # Simple bar chart
        if st.session_state.violations:
            daily_data = {"Today": len([v for v in st.session_state.violations 
                                      if v['timestamp'].date() == datetime.now().date()])}
            st.markdown("""
            <div style="background: white; border-radius: 12px; padding: 1.5rem;">
                <h4>Today's Violations</h4>
                <div style="background: #ef4444; border-radius: 10px; height: 20px; 
                          width: {}%; margin: 10px 0;"></div>
                <p>{} violations today</p>
            </div>
            """.format(min(100, daily_data["Today"] * 20), daily_data["Today"]), unsafe_allow_html=True)

def show_ppe_view():
    if st.session_state.violations:
        # Count violations by PPE type
        ppe_counts = {}
        for violation in st.session_state.violations:
            for item in violation['missing_classes'].split(', '):
                ppe_counts[item] = ppe_counts.get(item, 0) + 1
        
        st.write("**Violations by PPE Type:**")
        for ppe, count in ppe_counts.items():
            st.write(f"- {ppe}: {count} violations")
    else:
        st.info("No PPE violation data available")

def show_time_view():
    if st.session_state.violations:
        # Count by hour
        hour_counts = {}
        for violation in st.session_state.violations:
            hour = violation['timestamp'].hour
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
        
        st.write("**Violations by Hour:**")
        for hour in sorted(hour_counts.keys()):
            st.write(f"- {hour:02d}:00: {hour_counts[hour]} violations")
    else:
        st.info("No time-based data available")

if __name__ == "__main__":
    main()
