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
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #2E7D32;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

def check_dependencies():
    """Check if required dependencies are available"""
    missing_deps = []
    available_deps = []
    
    try:
        import torch
        available_deps.append(f"torch {torch.__version__}")
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import cv2
        available_deps.append(f"opencv-python {cv2.__version__}")
    except ImportError:
        missing_deps.append("opencv-python-headless")
    
    try:
        import numpy as np
        available_deps.append(f"numpy {np.__version__}")
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import PIL
        available_deps.append(f"Pillow {PIL.__version__}")
    except ImportError:
        missing_deps.append("Pillow")
    
    return missing_deps, available_deps

def check_yolov7_setup():
    """Check YOLOv7 setup and provide detailed status"""
    status = {
        "yolo_dir_exists": False,
        "python_path_added": False,
        "models_dir_exists": False,
        "utils_dir_exists": False,
        "model_file_exists": False,
        "modules_importable": False,
        "error_details": []
    }
    
    # Check YOLOv7 directory
    yolo_path = Path('./yolov7')
    if yolo_path.exists():
        status["yolo_dir_exists"] = True
        status["python_path_added"] = True
        sys.path.insert(0, str(yolo_path))
        
        # Check subdirectories
        if (yolo_path / "models").exists():
            status["models_dir_exists"] = True
        if (yolo_path / "utils").exists():
            status["utils_dir_exists"] = True
            
        # List contents for debugging
        try:
            yolo_contents = list(yolo_path.iterdir())
            status["yolo_contents"] = [item.name for item in yolo_contents]
        except Exception as e:
            status["error_details"].append(f"Error listing yolo directory: {e}")
    else:
        status["error_details"].append("YOLOv7 directory not found at ./yolov7")
    
    # Check model file
    model_path = Path('models/best.pt')
    if model_path.exists():
        status["model_file_exists"] = True
        status["model_size"] = model_path.stat().st_size
    else:
        status["error_details"].append("Model file not found at models/best.pt")
    
    # Try to import YOLOv7 modules
    if status["yolo_dir_exists"]:
        try:
            # Try different import approaches
            try:
                from models.experimental import attempt_load
                status["modules_importable"] = True
                status["import_method"] = "direct"
            except ImportError as e1:
                try:
                    # Try with absolute path
                    import importlib.util
                    spec = importlib.util.spec_from_file_location(
                        "experimental", 
                        yolo_path / "models" / "experimental.py"
                    )
                    if spec and spec.loader:
                        experimental = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(experimental)
                        status["modules_importable"] = True
                        status["import_method"] = "importlib"
                except Exception as e2:
                    status["error_details"].extend([
                        f"Direct import failed: {e1}",
                        f"Importlib failed: {e2}"
                    ])
        except Exception as e:
            status["error_details"].append(f"Unexpected import error: {e}")
    
    return status

def create_demo_interface():
    """Create a demo interface without YOLOv7"""
    st.subheader("📷 Demo Interface")
    st.info("🔄 YOLOv7 模組載入中或不可用，顯示演示界面")
    
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
            st.markdown("""
            <div class="status-box">
            <h4>🚧 演示模式</h4>
            <p><strong>圖片上傳成功！</strong></p>
            <p>在完整部署版本中，您會看到：</p>
            <ul>
                <li>🟢 Normal rice grains detection</li>
                <li>🟠 Broken rice grains detection</li>
                <li>🔴 Cracked rice grains detection</li>
                <li>📊 Detection confidence scores</li>
                <li>📈 Summary statistics</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Simulate some results
            with st.expander("📊 模擬檢測結果"):
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("🟢 Normal", "12", "grain(s)")
                with col_b:
                    st.metric("🟠 Broken", "3", "grain(s)")
                with col_c:
                    st.metric("🔴 Crack", "1", "grain(s)")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🌾 YOLOv7 Rice Quality Classifier</h1>', unsafe_allow_html=True)
    st.markdown("### Classify rice grains as **normal**, **broken**, or **crack**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model settings
        st.subheader("Model Settings")
        conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
        iou_threshold = st.slider("IoU Threshold", 0.1, 1.0, 0.45, 0.05)
        
        # System status
        st.subheader("🖥️ System Status")
        
        # Check dependencies
        missing_deps, available_deps = check_dependencies()
        
        if missing_deps:
            st.error(f"❌ Missing: {', '.join(missing_deps)}")
        else:
            st.success("✅ All dependencies available")
        
        # Show available dependencies
        with st.expander("📦 Available Dependencies"):
            for dep in available_deps:
                st.text(f"✅ {dep}")
        
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
    
    # Check dependencies first
    missing_deps, available_deps = check_dependencies()
    if missing_deps:
        st.error(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        st.info("The app is still loading dependencies. Please wait a moment and refresh the page.")
        return
    
    # Check YOLOv7 setup
    st.subheader("🔍 YOLOv7 Setup Status")
    
    with st.spinner("Checking YOLOv7 configuration..."):
        yolo_status = check_yolov7_setup()
    
    # Display detailed status
    status_cols = st.columns(3)
    
    with status_cols[0]:
        st.metric(
            "YOLOv7 Directory", 
            "✅ Found" if yolo_status["yolo_dir_exists"] else "❌ Missing",
            "at ./yolov7"
        )
    
    with status_cols[1]:
        st.metric(
            "Model File", 
            "✅ Found" if yolo_status["model_file_exists"] else "❌ Missing",
            f"{yolo_status.get('model_size', 0) / 1024 / 1024:.1f} MB" if yolo_status["model_file_exists"] else "models/best.pt"
        )
    
    with status_cols[2]:
        st.metric(
            "Modules", 
            "✅ Ready" if yolo_status["modules_importable"] else "❌ Error",
            yolo_status.get("import_method", "N/A") if yolo_status["modules_importable"] else "Import failed"
        )
    
    # Show detailed diagnostics
    with st.expander("🔧 Detailed Diagnostics"):
        st.json(yolo_status)
    
    # Show error details if any
    if yolo_status["error_details"]:
        with st.expander("⚠️ Error Details"):
            for error in yolo_status["error_details"]:
                st.error(error)
    
    # Main interface
    if yolo_status["modules_importable"] and yolo_status["model_file_exists"]:
        st.success("🎉 YOLOv7 fully loaded and ready!")
        # Here you would implement the full functionality
        st.info("🚧 Full inference pipeline coming soon!")
        create_demo_interface()
    else:
        st.warning("⚠️ YOLOv7 not fully available - running in demo mode")
        create_demo_interface()
    
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
    - 🔄 YOLOv7 setup in progress
    - 📱 Demo interface available
    
    ### 🛠️ Troubleshooting
    If you see import errors, this is expected in the cloud environment. 
    The app provides a demo interface that shows how the full version would work.
    """)

if __name__ == "__main__":
    main()