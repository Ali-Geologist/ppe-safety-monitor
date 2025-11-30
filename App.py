import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import requests

# Set page config for mobile
st.set_page_config(
    page_title="SafetyEagle AI",
    page_icon="🦅",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Add custom CSS for mobile optimization
st.markdown("""
<style>
    .main > div {
        padding-top: 1rem;
    }
    .stCameraInput > div > div {
        border-radius: 10px;
    }
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'classes_available' not in st.session_state:
    st.session_state.classes_available = False

def load_model():
    """Load YOLO model with error handling"""
    try:
        model = YOLO('model.pt')
        
        # Check if model has classes
        if hasattr(model, 'names') and model.names:
            st.session_state.classes_available = True
            st.session_state.classes = list(model.names.values())
        else:
            st.session_state.classes_available = False
            st.session_state.classes = []
            
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def download_model_if_needed():
    """Download model if not present (for Streamlit Cloud)"""
    if not os.path.exists('model.pt'):
        st.warning("Model file not found. Please upload a model file or provide a download URL.")
        return False
    return True

def process_image(image, model, confidence_threshold=0.5):
    """Process image with YOLO model"""
    try:
        # Run inference
        results = model(image, conf=confidence_threshold)
        
        # Get annotated image
        annotated_image = results[0].plot()
        
        # Get detection information
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls]
                    detections.append({
                        'label': label,
                        'confidence': conf,
                        'class': cls
                    })
        
        return annotated_image, detections
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return image, []

# Main app
def main():
    st.title("🦅 SafetyEagle AI")
    st.markdown("### Security Above Safety Standards")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Model management
        st.subheader("Model Settings")
        if st.button("Load/Reload Model"):
            if download_model_if_needed():
                st.session_state.model = load_model()
                if st.session_state.model:
                    st.success("Model loaded successfully!")
        
        # Confidence threshold
        confidence = st.slider(
            "Detection Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1
        )
        
        # Class selection (if available)
        if st.session_state.classes_available:
            st.subheader("Detection Classes")
            selected_classes = st.multiselect(
                "Select classes to detect:",
                options=st.session_state.classes,
                default=st.session_state.classes
            )
        else:
            st.info("No classes available from the model")
            selected_classes = []

    # Main content area
    st.header("Mobile Camera Detection")
    
    # Mobile usage instructions
    with st.expander("📱 Mobile Usage Instructions"):
        st.markdown("""
        1. **Allow camera permissions** when prompted by your browser
        2. **Choose detection method** below (Live Camera or Upload)
        3. **Point your camera** at the area to analyze
        4. **View results** in real-time or after processing
        
        *Note: Works best on Chrome, Safari, and Firefox mobile browsers*
        """)

    # Detection method selection
    detection_method = st.radio(
        "Choose detection method:",
        ["📸 Live Camera", "📁 Upload Image", "🎥 Upload Video"],
        horizontal=True
    )

    # Initialize model if not loaded
    if st.session_state.model is None:
        if download_model_if_needed():
            st.session_state.model = load_model()

    if st.session_state.model is None:
        st.error("⚠️ Model not loaded. Please check if 'model.pt' exists and click 'Load/Reload Model' in the sidebar.")
        return

    # Live Camera Detection
    if detection_method == "📸 Live Camera":
        st.subheader("Live Camera Detection")
        
        # Camera input
        camera_image = st.camera_input(
            "Take a picture with your camera",
            help="Point your camera at the scene you want to analyze"
        )
        
        if camera_image is not None:
            # Convert to OpenCV format
            bytes_data = camera_image.getvalue()
            image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            # Process image
            with st.spinner("🔍 Analyzing image..."):
                processed_image, detections = process_image(image, st.session_state.model, confidence)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(camera_image, use_column_width=True)
            
            with col2:
                st.subheader("Detection Results")
                st.image(processed_image, use_column_width=True)
            
            # Display detection information
            if detections:
                st.subheader("📊 Detection Summary")
                for i, detection in enumerate(detections):
                    st.write(f"**{i+1}. {detection['label']}** - Confidence: {detection['confidence']:.2f}")
            else:
                st.info("No objects detected with the current confidence threshold.")

    # Image Upload Detection
    elif detection_method == "📁 Upload Image":
        st.subheader("Upload Image for Detection")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image or take a photo (on mobile)"
        )
        
        if uploaded_file is not None:
            # Convert to OpenCV format
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Display original image
            st.subheader("Original Image")
            st.image(image, channels="BGR", use_column_width=True)
            
            # Process image
            if st.button("Run Detection", type="primary"):
                with st.spinner("🔍 Analyzing image..."):
                    processed_image, detections = process_image(image, st.session_state.model, confidence)
                
                # Display results
                st.subheader("Detection Results")
                st.image(processed_image, channels="BGR", use_column_width=True)
                
                # Display detection information
                if detections:
                    st.subheader("📊 Detection Summary")
                    for i, detection in enumerate(detections):
                        st.write(f"**{i+1}. {detection['label']}** - Confidence: {detection['confidence']:.2f}")
                else:
                    st.info("No objects detected with the current confidence threshold.")

    # Video Upload Detection
    elif detection_method == "🎥 Upload Video":
        st.subheader("Upload Video for Detection")
        
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file for analysis"
        )
        
        if uploaded_video is not None:
            # Save uploaded video to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_video.read())
                video_path = tmp_file.name
            
            # Display original video
            st.subheader("Original Video")
            st.video(uploaded_video)
            
            if st.button("Process Video", type="primary"):
                st.warning("⚠️ Video processing is computationally intensive and may take several minutes.")
                
                # Process video (simplified - for full implementation you'd need video processing)
                st.info("Video processing feature requires additional implementation for frame-by-frame analysis.")
                
                # Clean up
                os.unlink(video_path)

    # Footer
    st.markdown("---")
    st.markdown(
        "🦅 **SafetyEagle AI** - Enhancing security through AI-powered computer vision | "
        "Built with Streamlit, OpenCV, and YOLO"
    )

if __name__ == "__main__":
    main()
