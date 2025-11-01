#!/usr/bin/env python3
"""
YOLOv7 Rice Quality Classification Streamlit App - Cloud Optimized Version
Supports image upload and classification
Classes: normal, broken, crack
"""

import os
import sys
import streamlit as st
from pathlib import Path
import tempfile
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(
    page_title="🌾 YOLOv7 Rice Quality Classifier",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def check_dependencies():
    """Check if required dependencies are available"""
    missing_deps = []
    
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python-headless")
    
    return missing_deps

def load_model_if_available():
    """Try to load the model if available"""
    try:
        # Add YOLOv7 to path
        yolo_path = Path('./yolov7')
        if yolo_path.exists():
            sys.path.append(str(yolo_path))
        
        # Check for model file
        model_path = Path('models/best.pt')
        if not model_path.exists():
            return None, "Model file not found. Please upload your best.pt file."
        
        # Try to import YOLOv7 modules
        try:
            from models.experimental import attempt_load
            from utils.general import check_img_size, non_max_suppression, scale_coords
            from utils.plots import plot_one_box
            from utils.torch_utils import select_device
            import torch
            
            # Load model
            device = select_device('')
            model = attempt_load(str(model_path), map_location=device)
            model.eval()
            
            return model, "Model loaded successfully!"
            
        except ImportError as e:
            return None, f"YOLOv7 modules not available: {e}"
        except Exception as e:
            return None, f"Error loading model: {e}"
            
    except Exception as e:
        return None, f"Unexpected error: {e}"

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🌾 YOLOv7 Rice Quality Classifier</h1>', unsafe_allow_html=True)
    st.markdown("### Classify rice grains as **normal**, **broken**, or **crack**")
    
    # Check dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        st.error(f"Missing dependencies: {', '.join(missing_deps)}")
        st.info("The app is still loading dependencies. Please wait a moment and refresh the page.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model settings
        st.subheader("Model Settings")
        conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
        iou_threshold = st.slider("IoU Threshold", 0.1, 1.0, 0.45, 0.05)
        
        # Model info
        st.subheader("📊 Model Information")
        st.info("""
        **Architecture**: YOLOv7
        **Classes**: 3 (normal, broken, crack)
        **Input Size**: 640x640
        **Format**: PyTorch (.pt)
        """)
        
        # Legend
        st.subheader("🏷️ Class Legend")
        st.markdown("""
        - 🟢 **Normal**: Healthy rice grains
        - 🟠 **Broken**: Damaged/broken grains
        - 🔴 **Crack**: Grains with cracks
        """)
    
    # Try to load model
    with st.spinner("Loading YOLOv7 model..."):
        model, status_msg = load_model_if_available()
    
    if model is None:
        st.warning(f"⚠️ {status_msg}")
        
        # Show model upload option
        st.subheader("📤 Model Upload")
        st.info("Since the model couldn't be loaded automatically, you can:")
        st.markdown("""
        1. **For local testing**: Place your `best.pt` file in the `models/` directory
        2. **For deployment**: Ensure your model is properly uploaded to GitHub with Git LFS
        3. **Alternative**: Upload your model file below (feature coming soon)
        """)
        
        # Placeholder for future model upload functionality
        uploaded_model = st.file_uploader(
            "Upload your best.pt model file (Coming Soon)",
            type=['pt'],
            disabled=True,
            help="This feature will be available in a future update"
        )
        
        # Show demo interface even without model
        st.subheader("📷 Demo Interface")
        st.info("This is how the interface would look with a loaded model:")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image of rice grains for quality classification"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                st.subheader("Detection Results")
                st.info("🔄 Model not loaded - results would appear here")
                st.markdown("""
                **With a loaded model, you would see:**
                - Detected rice grains with bounding boxes
                - Classification results (normal/broken/crack)
                - Confidence scores for each detection
                - Summary statistics
                """)
        
        return
    
    # Model loaded successfully
    st.success(f"✅ {status_msg}")
    
    # Main content
    st.subheader("📷 Image Classification")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image of rice grains for quality classification"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.subheader("Detection Results")
            
            with st.spinner("🔍 Analyzing image..."):
                # Here would be the actual inference code
                # For now, show a placeholder
                st.info("🚧 Inference functionality will be implemented after successful deployment")
                st.markdown("""
                **Processing steps:**
                1. ✅ Image uploaded successfully
                2. ✅ Model loaded and ready
                3. 🔄 Running YOLOv7 inference...
                4. ⏳ Processing detections...
                5. ⏳ Generating results...
                """)
    
    # Instructions
    st.subheader("ℹ️ How to Use")
    st.markdown("""
    ### 🚀 Quick Start
    1. **Upload Image**: Use the file uploader to select an image of rice grains
    2. **Adjust Settings**: Use the sidebar to fine-tune detection parameters
    3. **View Results**: See detected rice grains with classifications
    
    ### 📊 Understanding Results
    - **Normal (🟢)**: Healthy, intact rice grains
    - **Broken (🟠)**: Damaged or broken rice grains
    - **Crack (🔴)**: Rice grains with visible cracks
    
    ### 🔧 Current Status
    - ✅ Streamlit app deployed successfully
    - ✅ Dependencies loaded
    - ✅ Model architecture ready
    - 🔄 Full inference pipeline coming soon
    """)

if __name__ == "__main__":
    main()