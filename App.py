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
    if 'project_info' not in st.session_state:
        st.session_state.project_info = {
            'company_name': '',
            'project_name': '',
            'engineer_name': '',
            'work_type': 'Offshore Operations',
            'workers_assigned': 0,
            'project_hours': 0,
        }

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
    color = "#22c55e" if percentage >= 90 else "#f59e0b" if percentage >= 80 else "#ef4444"
    
    gauge_html = f"""
    <div class="chart-container" style="text-align: center;">
        <h4>{title}</h4>
        <div style="position: relative; width: 150px; height: 150px; margin: 0 auto;">
            <div style="position: absolute; top: 0; left: 0; width: 150px; height: 150px; 
                      border-radius: 50%; background: conic-gradient({color} 0% {percentage}%, #e5e7eb {percentage}% 100%);">
            </div>
            <div style="position: absolute; top: 15px; left: 15px; width: 120px; height: 120px; 
                      border-radius: 50%; background: white; display: flex; align-items: center; justify-content: center;">
                <span style="font-size: 2rem; font-weight: bold; color: {color};">{percentage}%</span>
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
        return {}, {}
    
    # PPE violation distribution
    ppe_violations = {}
    time_violations = {}
    
    for violation in st.session_state.violations:
        # Count by PPE type
        missing_items = violation.get('missing_classes', 'Unknown').split(', ')
        for item in missing_items:
            ppe_violations[item] = ppe_violations.get(item, 0) + 1
        
        # Count by hour of day
        hour = violation['timestamp'].hour
        time_slot = f"{hour:02d}:00-{hour+1:02d}:00"
        time_violations[time_slot] = time_violations.get(time_slot, 0) + 1
    
    return ppe_violations, time_violations

def analyze_image_for_ppe(image):
    """Simulate PPE detection analysis on the captured image"""
    # This is a simulation - in a real app, you'd use a trained model here
    import random
    
    # Simulate AI analysis with random results for demonstration
    detected_ppe = []
    missing_ppe = []
    
    for ppe_id, ppe_data in st.session_state.selected_ppe.items():
        # 80% chance of detecting each selected PPE item
        if random.random() < 0.8:
            detected_ppe.append(ppe_data)
        else:
            missing_ppe.append(ppe_data)
    
    return detected_ppe, missing_ppe

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
    tabs = st.tabs(["🏠 **Dashboard**", "📹 **Live Monitoring**", "⚙️ **Configuration**", "📊 **Analytics**"])
    
    with tabs[0]:
        show_dashboard()
    with tabs[1]:
        show_live_monitoring()
    with tabs[2]:
        show_configuration()
    with tabs[3]:
        show_analytics()

def show_dashboard():
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="modern-card"><div class="card-header">📹 Live Camera Feed</div></div>', unsafe_allow_html=True)
        
        # Show actual camera image if available
        if st.session_state.camera_image is not None:
            st.image(st.session_state.camera_image, 
                    caption="Latest Camera Capture - PPE Analysis Ready", 
                    use_column_width=True)
            
            # Show analysis results if we have a recent image
            st.info("🔄 Ready for PPE analysis. Go to Live Monitoring to analyze this image.")
        elif st.session_state.mobile_camera_active:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                        border-radius: 12px; 
                        height: 400px; 
                        display: flex; 
                        align-items: center; 
                        justify-content: center; 
                        color: white;
                        margin-bottom: 1rem;">
                <div style="text-align: center;">
                    <div style="font-size: 4rem; margin-bottom: 1rem;">📱</div>
                    <h3>Mobile Camera Active</h3>
                    <p>Go to Live Monitoring to capture images</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #f8fafc; 
                        border-radius: 12px; 
                        height: 400px; 
                        display: flex; 
                        align-items: center; 
                        justify-content: center; 
                        color: #6b7280;
                        border: 2px dashed #d1d5db;
                        margin-bottom: 1rem;">
                <div style="text-align: center;">
                    <div style="font-size: 4rem; margin-bottom: 1rem;">🔒</div>
                    <h3>Monitoring Inactive</h3>
                    <p>Start mobile camera to begin PPE detection</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Real-time stats based on actual data
        compliance_rate = calculate_compliance_rate()
        total_violations = len(st.session_state.violations)
        workers = st.session_state.project_info['workers_assigned'] or 0
        ppe_items = len(st.session_state.selected_ppe)
        
        col1a, col1b, col1c = st.columns(3)
        with col1a:
            st.metric("Workers Monitored", workers, "Live" if st.session_state.monitoring else "Offline")
        with col1b:
            st.metric("PPE Items Tracked", ppe_items)
        with col1c:
            st.metric("Compliance Rate", f"{compliance_rate:.1f}%", f"{total_violations} violations")
    
    with col2:
        st.markdown('<div class="modern-card"><div class="card-header">🚨 Safety Alerts</div></div>', unsafe_allow_html=True)
        
        if st.session_state.violations:
            latest_violation = st.session_state.violations[-1]
            st.markdown(f"""
            <div class="alert-warning">
                <strong>⚠️ Recent Violation Detected</strong><br>
                {latest_violation.get('missing_classes', 'Unknown PPE')}<br>
                <small>{latest_violation['timestamp'].strftime('%Y-%m-%d %H:%M')}</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-success">
                <strong>✅ No Safety Violations</strong><br>
                All monitored PPE items are compliant
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<div class="modern-card"><div class="card-header">🔧 Quick Actions</div></div>', unsafe_allow_html=True)
        
        if st.button("📱 Go to Mobile Camera", use_container_width=True, type="primary"):
            st.switch_page("?tab=Live+Monitoring")

def show_live_monitoring():
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="modern-card"><div class="card-header">📱 Mobile Camera Live Monitoring</div></div>', unsafe_allow_html=True)
        
        # Mobile Camera Section
        st.subheader("📸 Capture Image for PPE Analysis")
        
        # Streamlit's camera input
        camera_image = st.camera_input(
            "Take a picture for PPE detection",
            help="Position camera to capture workers and their safety equipment clearly"
        )
        
        if camera_image is not None:
            # Store the image in session state
            st.session_state.camera_image = camera_image
            st.session_state.mobile_camera_active = True
            st.session_state.monitoring = True
            
            # Display the captured image
            st.image(camera_image, caption="Captured Image - Ready for Analysis", use_column_width=True)
            
            # PPE Analysis Section
            st.subheader("🔍 PPE Detection Analysis")
            
            if st.button("🎯 Analyze PPE Compliance", type="primary", use_container_width=True):
                with st.spinner("Analyzing image for PPE compliance..."):
                    # Simulate processing time
                    time.sleep(2)
                    
                    # Analyze the image for PPE
                    detected_ppe, missing_ppe = analyze_image_for_ppe(camera_image)
                    
                    # Display results
                    col_result1, col_result2 = st.columns(2)
                    
                    with col_result1:
                        st.success("✅ Detected PPE Items:")
                        for ppe in detected_ppe:
                            st.write(f"- {ppe}")
                    
                    with col_result2:
                        if missing_ppe:
                            st.error("❌ Missing PPE Items:")
                            for ppe in missing_ppe:
                                st.write(f"- {ppe}")
                            
                            # Auto-report violation
                            violation = {
                                'timestamp': datetime.now(),
                                'missing_classes': ', '.join(missing_ppe),
                                'worker_id': 'Camera Detection',
                                'source': 'Mobile Camera AI',
                                'image': camera_image
                            }
                            st.session_state.violations.append(violation)
                            st.error(f"🚨 Violation reported: {', '.join(missing_ppe)}")
                        else:
                            st.success("🎉 All required PPE items detected!")
            
            # Manual violation reporting as backup
            st.subheader("📝 Manual PPE Check")
            st.info("Use this if automatic detection needs correction")
            
            col_manual1, col_manual2 = st.columns(2)
            with col_manual1:
                missing_ppe_manual = st.multiselect(
                    "Manually report missing PPE:",
                    [ppe["name"] for ppe in PPE_CLASSES.values()],
                    help="Select any missing PPE items you observe"
                )
            with col_manual2:
                worker_id_manual = st.text_input("Worker ID:", placeholder="W001")
            
            if st.button("🚨 Report Manual Violation", type="secondary"):
                if missing_ppe_manual:
                    violation = {
                        'timestamp': datetime.now(),
                        'missing_classes': ', '.join(missing_ppe_manual),
                        'worker_id': worker_id_manual or 'Unknown',
                        'source': 'Manual Report',
                        'image': camera_image
                    }
                    st.session_state.violations.append(violation)
                    st.error(f"Violation reported: {', '.join(missing_ppe_manual)}")
                else:
                    st.warning("Please select missing PPE items")
        
        else:
            # Camera instructions when no image is captured
            st.info("""
            ### 📋 Mobile Camera Instructions:
            
            1. **Click the camera button above** to activate your device camera
            2. **Allow camera permissions** when prompted by your browser
            3. **Position your device** to monitor the work area
            4. **Capture clear images** of workers and their safety equipment
            5. **Click 'Analyze PPE Compliance'** to check for violations
            
            ### 🎯 Best Practices:
            - Capture from 3-5 meters distance
            - Ensure good lighting conditions
            - Include full body view of workers
            - Avoid blurry or dark images
            - Position at eye level for best angle
            """)
            
            # Show camera status
            if st.session_state.mobile_camera_active:
                st.success("✅ Mobile camera is ready - waiting for image capture")
            else:
                st.warning("📱 Camera not active - click the camera button to start")
    
    with col2:
        st.markdown('<div class="modern-card"><div class="card-header">⚙️ Detection Settings</div></div>', unsafe_allow_html=True)
        
        confidence = st.slider(
            "Detection Confidence",
            min_value=0.1,
            max_value=0.9,
            value=st.session_state.detection_settings['confidence'],
            step=0.1,
            help="Higher values = more accurate but fewer detections"
        )
        
        processing_speed = st.selectbox(
            "Processing Speed",
            ["Fast", "Balanced", "High Accuracy"],
            index=1
        )
        
        # Update settings
        st.session_state.detection_settings['confidence'] = confidence
        
        st.markdown('<div class="modern-card"><div class="card-header">🎯 Active PPE Detection</div></div>', unsafe_allow_html=True)
        
        # PPE selection
        st.subheader("Selected PPE Items")
        selected_count = 0
        for ppe_id, ppe_data in PPE_CLASSES.items():
            is_selected = st.checkbox(
                f"{ppe_data['icon']} {ppe_data['name']}", 
                value=ppe_id in st.session_state.selected_ppe,
                key=f"monitor_{ppe_id}"
            )
            if is_selected:
                st.session_state.selected_ppe[ppe_id] = ppe_data['name']
                selected_count += 1
            elif ppe_id in st.session_state.selected_ppe:
                del st.session_state.selected_ppe[ppe_id]
        
        st.info(f"Monitoring {selected_count} PPE items")
        
        # Quick actions
        st.markdown('<div class="modern-card"><div class="card-header">🔧 Quick Actions</div></div>', unsafe_allow_html=True)
        
        if st.session_state.camera_image is not None:
            if st.button("🔄 Clear Current Image", use_container_width=True):
                st.session_state.camera_image = None
                st.rerun()

def show_configuration():
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="modern-card"><div class="card-header">🏢 Project Setup</div></div>', unsafe_allow_html=True)
        
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
        
        st.subheader("Select PPE to Monitor")
        
        for ppe_id, ppe_data in PPE_CLASSES.items():
            col_ppe1, col_ppe2 = st.columns([1, 4])
            with col_ppe1:
                st.write(f"**{ppe_data['icon']}**")
            with col_ppe2:
                if st.checkbox(ppe_data['name'], value=ppe_id in st.session_state.selected_ppe, key=f"ppe_{ppe_id}"):
                    st.session_state.selected_ppe[ppe_id] = ppe_data['name']
        
        if st.button("✅ Apply PPE Selection", use_container_width=True, type="primary"):
            st.success(f"Now monitoring {len(st.session_state.selected_ppe)} PPE items")

def show_analytics():
    st.markdown('<div class="modern-card"><div class="card-header">📊 Safety Analytics Dashboard</div></div>', unsafe_allow_html=True)
    
    # Calculate real metrics
    compliance_rate = calculate_compliance_rate()
    total_violations = len(st.session_state.violations)
    workers = st.session_state.project_info['workers_assigned'] or 1
    ppe_violations, time_violations = get_violation_stats()
    
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
    
    if st.session_state.violations:
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # Weekly compliance trend (simplified)
            weekly_data = {}
            for violation in st.session_state.violations:
                week = violation['timestamp'].strftime("Week %U")
                weekly_data[week] = weekly_data.get(week, 0) + 1
            
            # Convert violations to compliance rates
            compliance_trend = {}
            for week in weekly_data:
                # Simple calculation: assume 20 workers and reduce compliance based on violations
                base_compliance = 95
                weekly_violations = weekly_data[week]
                compliance_trend[week] = max(60, base_compliance - (weekly_violations * 2))
            
            st.markdown(create_simple_bar_chart(compliance_trend, "Weekly Compliance Trend", "#2563eb"))
        
        with col_chart2:
            # Compliance gauge
            st.markdown(create_compliance_gauge(int(compliance_rate), "Overall Compliance Rate"))
        
        # PPE violation distribution
        st.markdown('<div class="modern-card"><div class="card-header">📋 PPE Violation Distribution</div></div>', unsafe_allow_html=True)
        
        col_viol1, col_viol2 = st.columns(2)
        
        with col_viol1:
            st.markdown(create_simple_bar_chart(ppe_violations, "Violations by PPE Type", "#ef4444"))
        
        with col_viol2:
            st.markdown(create_simple_bar_chart(time_violations, "Violations by Time of Day", "#f59e0b"))
    
    # Recent alerts table
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

def generate_safety_report():
    """Generate a comprehensive safety report based on actual data"""
    compliance_rate = calculate_compliance_rate()
    total_violations = len(st.session_state.violations)
    ppe_violations, time_violations = get_violation_stats()
    
    report = f"""
    SAFETYEAGLE AI - SAFETY MONITORING REPORT
    =========================================
    
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    
    PROJECT INFORMATION
    -------------------
    Company: {st.session_state.project_info['company_name'] or 'Not specified'}
    Project: {st.session_state.project_info['project_name'] or 'Not specified'}
    Safety Engineer: {st.session_state.project_info['engineer_name'] or 'Not specified'}
    Work Type: {st.session_state.project_info['work_type']}
    Workers: {st.session_state.project_info['workers_assigned']}
    
    SAFETY PERFORMANCE SUMMARY
    --------------------------
    Overall Compliance Rate: {compliance_rate:.1f}%
    Total Violations: {total_violations}
    PPE Items Monitored: {len(st.session_state.selected_ppe)}
    Monitoring Period: Since first violation recorded
    
    VIOLATION ANALYSIS
    ------------------
    """
    
    if ppe_violations:
        report += "PPE Violation Distribution:\n"
        for ppe, count in ppe_violations.items():
            report += f"  - {ppe}: {count} violations\n"
    else:
        report += "No PPE violations recorded.\n"
    
    report += f"""
    
    RECOMMENDATIONS
    ---------------
    """
    
    if compliance_rate >= 95:
        report += "1. Continue current excellent safety protocols\n"
    elif compliance_rate >= 80:
        report += "1. Conduct refresher PPE training sessions\n"
    else:
        report += "1. Implement immediate safety intervention program\n"
    
    report += "2. Regular safety equipment inspections\n"
    report += "3. Ongoing worker safety awareness programs\n"
    report += "4. Continuous monitoring and improvement\n"
    
    report += f"""
    
    MONITORED PPE ITEMS
    -------------------
    {', '.join(st.session_state.selected_ppe.values()) if st.session_state.selected_ppe else 'No PPE items selected'}
    
    ---
    Generated by SafetyEagle AI
    Confidential Safety Report
    """
    return report

if __name__ == "__main__":
    main()
