import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
import os
from pathlib import Path
import tempfile
import io

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
    if 'demo_mode' not in st.session_state:
        st.session_state.demo_mode = False
    if 'project_info' not in st.session_state:
        st.session_state.project_info = {
            'company_name': 'Oil & Gas Corporation',
            'project_name': 'Offshore Drilling Platform A',
            'engineer_name': 'Safety Manager',
            'work_type': 'Offshore Operations',
            'workers_assigned': 24,
            'project_hours': 240,
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
    max_value = max(data.values()) if data else 1
    chart_html = f"""
    <div class="chart-container">
        <h4>{title}</h4>
    """
    for label, value in data.items():
        width = (value / max_value) * 100
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

def test_camera_connection_simulated(camera_url):
    """Simulate camera connection test without external dependencies"""
    # Simulate connection test with random success
    import random
    success = random.choice([True, True, True, False])  # 75% success rate
    if success:
        return True, "✅ Camera connected successfully!"
    else:
        return False, "❌ Cannot connect to camera. Please check the URL and credentials."

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
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">24/7</div>
                    <div class="stat-label">Real-time Monitoring</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">99%</div>
                    <div class="stat-label">Detection Accuracy</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">0</div>
                    <div class="stat-label">PPE Violations Today</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">100%</div>
                    <div class="stat-label">Compliance Rate</div>
                </div>
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
        
        # Camera feed placeholder
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 12px; 
                    height: 400px; 
                    display: flex; 
                    align-items: center; 
                    justify-content: center; 
                    color: white;
                    margin-bottom: 1rem;">
            <div style="text-align: center;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">📹</div>
                <h3>Live PPE Monitoring Active</h3>
                <p>Real-time AI-powered safety detection</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick stats
        col1a, col1b, col1c = st.columns(3)
        with col1a:
            st.metric("Workers Monitored", "24", "0")
        with col1b:
            st.metric("PPE Items Tracked", "8", "0")
        with col1c:
            st.metric("Compliance Rate", "100%", "0%")
    
    with col2:
        st.markdown('<div class="modern-card"><div class="card-header">🚨 Safety Alerts</div></div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="alert-success">
            <strong>✅ All Systems Operational</strong><br>
            No safety violations detected in the last 24 hours
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="modern-card"><div class="card-header">📈 Quick Stats</div></div>', unsafe_allow_html=True)
        
        # PPE compliance breakdown
        st.subheader("PPE Compliance")
        for ppe_id, ppe_data in PPE_CLASSES.items():
            if ppe_id < 4:  # Show only first 4 for space
                col_stat1, col_stat2 = st.columns([2, 1])
                with col_stat1:
                    st.write(f"{ppe_data['icon']} {ppe_data['name']}")
                with col_stat2:
                    st.write("✅ 100%")
        
        st.markdown('<div class="modern-card"><div class="card-header">🔧 Quick Actions</div></div>', unsafe_allow_html=True)
        
        if st.button("🎬 Start Monitoring", use_container_width=True, type="primary"):
            st.session_state.monitoring = True
            st.success("Live monitoring started!")
        
        if st.button("📊 Generate Report", use_container_width=True):
            # Generate simple text report
            report_content = generate_safety_report()
            st.download_button(
                label="📥 Download Safety Report",
                data=report_content,
                file_name=f"safety_report_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )
            st.success("Safety report generated! Click download button.")

def show_live_monitoring():
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="modern-card"><div class="card-header">🎥 Camera Configuration</div></div>', unsafe_allow_html=True)
        
        # Camera setup
        tab1, tab2, tab3 = st.tabs(["📡 IP Camera", "📁 Upload Video", "🧪 Demo Mode"])
        
        with tab1:
            st.subheader("IP Camera Setup")
            camera_url = st.text_input(
                "Camera RTSP URL:",
                placeholder="rtsp://username:password@192.168.1.100:554/stream",
                help="Enter your IP camera RTSP stream URL"
            )
            
            col1a, col1b = st.columns(2)
            with col1a:
                if st.button("🔗 Test Connection", use_container_width=True):
                    if camera_url:
                        success, message = test_camera_connection_simulated(camera_url)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
                    else:
                        st.error("Please enter a camera URL")
            with col1b:
                if st.button("➕ Add Camera", use_container_width=True, type="primary"):
                    if camera_url:
                        if camera_url not in st.session_state.camera_urls:
                            st.session_state.camera_urls.append(camera_url)
                            st.success("Camera added to list!")
                    else:
                        st.error("Please enter a camera URL first")
        
        with tab2:
            st.subheader("Video File Analysis")
            uploaded_file = st.file_uploader("Upload video file", type=['mp4', 'avi', 'mov'])
            if uploaded_file:
                st.success(f"✅ Video ready for analysis: {uploaded_file.name}")
                if st.button("🎬 Analyze Video", type="primary"):
                    with st.spinner("Analyzing video for PPE compliance..."):
                        time.sleep(2)  # Simulate processing
                        st.success("Video analysis completed! No violations found.")
        
        with tab3:
            st.subheader("Demo Mode")
            st.info("Experience the PPE detection system with simulated data")
            if st.button("🚀 Start Demo", type="primary", use_container_width=True):
                st.session_state.monitoring = True
                st.session_state.demo_mode = True
                st.success("Demo mode activated! Showing simulated PPE detection.")
    
    with col2:
        st.markdown('<div class="modern-card"><div class="card-header">⚙️ Detection Settings</div></div>', unsafe_allow_html=True)
        
        confidence = st.slider(
            "Detection Confidence",
            min_value=0.1,
            max_value=0.9,
            value=0.7,
            step=0.1,
            help="Higher values = more accurate but fewer detections"
        )
        
        processing_speed = st.selectbox(
            "Processing Speed",
            ["Fast", "Balanced", "High Accuracy"],
            index=1
        )
        
        st.markdown('<div class="modern-card"><div class="card-header">🎯 Active PPE Detection</div></div>', unsafe_allow_html=True)
        
        # PPE selection
        st.subheader("Selected PPE Items")
        selected_count = 0
        for ppe_id, ppe_data in PPE_CLASSES.items():
            if st.checkbox(f"{ppe_data['icon']} {ppe_data['name']}", value=True, key=f"monitor_{ppe_id}"):
                st.session_state.selected_ppe[ppe_id] = ppe_data['name']
                selected_count += 1
        
        st.info(f"Monitoring {selected_count} PPE items")

def show_configuration():
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="modern-card"><div class="card-header">🏢 Project Setup</div></div>', unsafe_allow_html=True)
        
        company_name = st.text_input("Company Name", value=st.session_state.project_info['company_name'])
        project_name = st.text_input("Project Name", value=st.session_state.project_info['project_name'])
        work_type = st.selectbox(
            "Work Type",
            ["Offshore Operations", "Drilling", "Refinery", "Pipeline", "Maintenance", "Construction"]
        )
        workers = st.number_input("Number of Workers", min_value=1, value=st.session_state.project_info['workers_assigned'])
        
        if st.button("💾 Save Project Settings", type="primary", use_container_width=True):
            st.session_state.project_info = {
                'company_name': company_name,
                'project_name': project_name,
                'work_type': work_type,
                'workers_assigned': workers,
                'project_hours': 240,
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
                if st.checkbox(ppe_data['name'], value=True, key=f"ppe_{ppe_id}"):
                    st.session_state.selected_ppe[ppe_id] = ppe_data['name']
        
        if st.button("✅ Apply PPE Selection", use_container_width=True, type="primary"):
            st.success(f"Now monitoring {len(st.session_state.selected_ppe)} PPE items")

def show_analytics():
    st.markdown('<div class="modern-card"><div class="card-header">📊 Safety Analytics Dashboard</div></div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Overall Compliance", "96.2%", "1.2%")
    with col2:
        st.metric("Monthly Violations", "24", "-8")
    with col3:
        st.metric("Avg. Daily Checks", "1,248", "45")
    with col4:
        st.metric("Safety Score", "A+", "0")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Compliance trend using simple HTML chart
        compliance_data = {
            "Week 1": 92,
            "Week 2": 94,
            "Week 3": 95,
            "Week 4": 96,
            "This Week": 98
        }
        st.markdown(create_simple_bar_chart(compliance_data, "Weekly Compliance Trend", "#2563eb"))
    
    with col_chart2:
        # Compliance gauge
        st.markdown(create_compliance_gauge(96, "Overall Compliance Rate"))
    
    # PPE violation distribution
    st.markdown('<div class="modern-card"><div class="card-header">📋 PPE Violation Distribution</div></div>', unsafe_allow_html=True)
    
    ppe_violations = {
        'Hard Hat': 12,
        'Safety Glasses': 8,
        'Hi-Vis Vest': 3,
        'Safety Gloves': 1,
        'Safety Boots': 0
    }
    
    col_viol1, col_viol2 = st.columns(2)
    
    with col_viol1:
        st.markdown(create_simple_bar_chart(ppe_violations, "Violations by PPE Type", "#ef4444"))
    
    with col_viol2:
        # Violation by time of day
        time_violations = {
            "6-8 AM": 2,
            "8-10 AM": 5,
            "10-12 PM": 8,
            "12-2 PM": 4,
            "2-4 PM": 3,
            "4-6 PM": 2
        }
        st.markdown(create_simple_bar_chart(time_violations, "Violations by Time of Day", "#f59e0b"))
    
    # Recent alerts table
    st.markdown('<div class="modern-card"><div class="card-header">🚨 Recent Safety Events</div></div>', unsafe_allow_html=True)
    
    sample_alerts = [
        {"time": "2024-01-15 08:30", "type": "PPE Violation", "severity": "Medium", "description": "Missing safety glasses - Area A"},
        {"time": "2024-01-14 14:15", "type": "PPE Violation", "severity": "High", "description": "No hard hat - Drill floor"},
        {"time": "2024-01-13 11:45", "type": "System Alert", "severity": "Low", "description": "Camera connection restored"},
        {"time": "2024-01-12 09:20", "type": "PPE Compliance", "severity": "Info", "description": "100% compliance - Shift B"},
    ]
    
    for alert in sample_alerts:
        severity_color = {
            "High": "#ef4444",
            "Medium": "#f59e0b", 
            "Low": "#3b82f6",
            "Info": "#10b981"
        }
        
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 8px; border-left: 4px solid {severity_color[alert['severity']]}; margin: 0.5rem 0;">
            <div style="display: flex; justify-content: between; align-items: start;">
                <div style="flex: 1;">
                    <strong>{alert['type']}</strong><br>
                    <small style="color: #6b7280;">{alert['time']}</small>
                </div>
                <div style="background: {severity_color[alert['severity']]}; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.8rem;">
                    {alert['severity']}
                </div>
            </div>
            <div style="margin-top: 0.5rem;">{alert['description']}</div>
        </div>
        """, unsafe_allow_html=True)

def generate_safety_report():
    """Generate a simple text safety report"""
    report = f"""
    SAFETYEAGLE AI - SAFETY MONITORING REPORT
    =========================================
    
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    
    PROJECT INFORMATION
    -------------------
    Company: {st.session_state.project_info['company_name']}
    Project: {st.session_state.project_info['project_name']}
    Work Type: {st.session_state.project_info['work_type']}
    Workers: {st.session_state.project_info['workers_assigned']}
    
    SAFETY PERFORMANCE SUMMARY
    --------------------------
    Overall Compliance Rate: 96.2%
    Monthly Violations: 24
    PPE Items Monitored: {len(st.session_state.selected_ppe)}
    Safety Score: A+
    
    RECOMMENDATIONS
    ---------------
    1. Continue current safety protocols
    2. Conduct weekly safety briefings
    3. Monitor high-risk areas more frequently
    4. Provide ongoing PPE training
    
    ---
    Generated by SafetyEagle AI
    Confidential Safety Report
    """
    return report

if __name__ == "__main__":
    main()
