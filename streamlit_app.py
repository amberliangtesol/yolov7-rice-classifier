#!/usr/bin/env python3
"""
YOLOv7 Rice Quality Classification Streamlit App
Supports image upload, webcam capture, and video processing
Classes: normal, broken, crack
"""

import os
import sys
import time
import cv2
import torch
import numpy as np
import streamlit as st
from pathlib import Path
import tempfile
import shutil
from PIL import Image
import io

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
    .detection-box {
        border: 2px solid #4CAF50;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Add YOLOv7 to path
@st.cache_resource
def setup_yolo_path():
    yolo_path = Path('./yolov7')
    if yolo_path.exists():
        sys.path.append(str(yolo_path))
        return True
    return False

# Import YOLOv7 modules
def import_yolo_modules():
    try:
        from models.experimental import attempt_load
        from utils.general import check_img_size, non_max_suppression, scale_coords
        from utils.plots import plot_one_box
        from utils.torch_utils import select_device
        return attempt_load, check_img_size, non_max_suppression, scale_coords, plot_one_box, select_device
    except ImportError as e:
        st.error(f"YOLOv7 modules not found: {e}")
        st.error("Please make sure YOLOv7 is properly installed.")
        return None

class RiceClassifierStreamlit:
    def __init__(self, weights_path='models/best.pt', device='', img_size=640, conf_thres=0.25, iou_thres=0.45):
        """Initialize YOLOv7 Rice Classifier for Streamlit"""
        self.weights_path = weights_path
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # Class names for rice quality
        self.names = ['normal', 'broken', 'crack']
        self.colors = [(0, 255, 0), (255, 165, 0), (255, 0, 0)]  # Green, Orange, Red
        
        # Import YOLOv7 functions
        modules = import_yolo_modules()
        if modules is None:
            self.model = None
            return
            
        self.attempt_load, self.check_img_size, self.non_max_suppression, self.scale_coords, self.plot_one_box, self.select_device = modules
        
        # Initialize device
        self.device = self.select_device(device)
        
        # Load model
        self.model = self._load_model()
        
        # Create output directory
        self.output_dir = Path('runs/detect')
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @st.cache_resource
    def _load_model(_self):
        """Load YOLOv7 model with trained weights"""
        try:
            # Handle PyTorch 2.6+ weights_only security change
            import torch.serialization
            torch.serialization.add_safe_globals(['numpy._core.multiarray._reconstruct'])
            
            model = _self.attempt_load(_self.weights_path, map_location=_self.device)
            model.eval()
            
            # Check image size
            _self.img_size = _self.check_img_size(_self.img_size, s=model.stride.max())
            
            st.success(f"✅ Model loaded successfully!")
            st.info(f"Device: {_self.device} | Image size: {_self.img_size}")
            
            return model
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return None
    
    def preprocess_image(self, img):
        """Preprocess image for inference"""
        # Resize image
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize and convert to tensor
        img_tensor = torch.from_numpy(img_rgb).to(self.device)
        img_tensor = img_tensor.float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        
        return img_tensor, img_resized
    
    def predict_image(self, image):
        """Run inference on a single image"""
        if self.model is None:
            return None, "Model not loaded"
        
        try:
            # Convert PIL Image to OpenCV format
            if isinstance(image, Image.Image):
                img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                img = image
            
            original_img = img.copy()
            h, w = img.shape[:2]
            
            # Preprocess
            img_tensor, img_resized = self.preprocess_image(img)
            
            # Inference
            with torch.no_grad():
                pred = self.model(img_tensor, augment=False)[0]
                pred = self.non_max_suppression(pred, self.conf_thres, self.iou_thres)
            
            detection_results = []
            
            # Process detections
            for i, det in enumerate(pred):
                if len(det):
                    # Rescale boxes to original image size
                    det[:, :4] = self.scale_coords(img_resized.shape, det[:, :4], (h, w)).round()
                    
                    # Draw boxes and labels
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        color = self.colors[int(cls)]
                        
                        # Draw bounding box
                        self.plot_one_box(xyxy, original_img, label=label, color=color, line_thickness=2)
                        
                        # Store detection info
                        x1, y1, x2, y2 = [int(x) for x in xyxy]
                        detection_results.append({
                            'class': self.names[int(cls)],
                            'confidence': float(conf),
                            'bbox': [x1, y1, x2, y2]
                        })
            
            # Convert back to RGB for display
            result_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            
            return result_img_rgb, detection_results
            
        except Exception as e:
            return None, f"Error during inference: {e}"

# Initialize classifier
@st.cache_resource
def load_classifier():
    """Load the rice classifier"""
    if not setup_yolo_path():
        st.error("❌ YOLOv7 directory not found!")
        return None
    
    weights_path = 'models/best.pt'
    if not os.path.exists(weights_path):
        st.error(f"❌ Model file not found at {weights_path}")
        st.error("Please place your best.pt file in the models/ directory")
        return None
    
    classifier = RiceClassifierStreamlit(weights_path=weights_path)
    return classifier

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
    
    # Load classifier
    classifier = load_classifier()
    if classifier is None:
        st.error("Failed to load classifier. Please check your setup.")
        return
    
    # Update thresholds
    classifier.conf_thres = conf_threshold
    classifier.iou_thres = iou_threshold
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📷 Image Upload", "🎥 Video Upload", "📹 Webcam", "ℹ️ Instructions"])
    
    with tab1:
        st.header("📷 Image Classification")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image of rice grains for quality classification"
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                st.subheader("Detection Results")
                
                # Run prediction
                with st.spinner("🔍 Analyzing image..."):
                    result_img, detections = classifier.predict_image(image)
                
                if result_img is not None:
                    st.image(result_img, caption="Detection Results", use_column_width=True)
                    
                    # Display detection statistics
                    if detections:
                        st.success(f"✅ Detected {len(detections)} objects")
                        
                        # Count by class
                        class_counts = {}
                        for det in detections:
                            class_name = det['class']
                            class_counts[class_name] = class_counts.get(class_name, 0) + 1
                        
                        # Display metrics
                        st.subheader("📊 Detection Summary")
                        metric_cols = st.columns(3)
                        
                        colors = {'normal': '🟢', 'broken': '🟠', 'crack': '🔴'}
                        for i, (class_name, count) in enumerate(class_counts.items()):
                            with metric_cols[i % 3]:
                                st.metric(
                                    f"{colors.get(class_name, '⚪')} {class_name.capitalize()}",
                                    count
                                )
                        
                        # Detailed results
                        st.subheader("📋 Detailed Detections")
                        for i, det in enumerate(detections):
                            with st.expander(f"Detection {i+1}: {det['class'].capitalize()}"):
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.write(f"**Class**: {det['class']}")
                                    st.write(f"**Confidence**: {det['confidence']:.3f}")
                                with col_b:
                                    bbox = det['bbox']
                                    st.write(f"**Bounding Box**: ({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]})")
                    else:
                        st.warning("⚠️ No rice grains detected. Try adjusting the confidence threshold.")
                else:
                    st.error("❌ Failed to process image")
    
    with tab2:
        st.header("🎥 Video Classification")
        st.info("📹 Video processing feature is available but requires additional implementation for Streamlit.")
        st.markdown("""
        **For video processing, you can use the command line:**
        ```bash
        cd yolov7
        python detect.py --weights ../models/best.pt --source your_video.mp4 --conf 0.25
        ```
        """)
        
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov'],
            help="Upload a video of rice grains for batch classification"
        )
        
        if uploaded_video is not None:
            st.video(uploaded_video)
            st.warning("Video processing will be implemented in a future update.")
    
    with tab3:
        st.header("📹 Webcam Detection")
        st.info("🎥 Webcam feature requires running YOLOv7 directly from command line.")
        st.markdown("""
        **To use webcam detection:**
        ```bash
        cd yolov7
        python detect.py --weights ../models/best.pt --source 0 --conf 0.25
        ```
        
        This will open a window showing live webcam feed with rice grain detections.
        """)
        
        # Webcam placeholder
        camera_placeholder = st.empty()
        with camera_placeholder:
            st.image("https://via.placeholder.com/640x480/cccccc/000000?text=Webcam+Placeholder", 
                    caption="Webcam feature available via command line")
    
    with tab4:
        st.header("ℹ️ How to Use")
        
        st.markdown("""
        ### 🚀 Quick Start
        1. **Upload Image**: Use the "Image Upload" tab to analyze single images
        2. **Adjust Settings**: Use the sidebar to fine-tune detection parameters
        3. **View Results**: See detected rice grains with bounding boxes and classifications
        
        ### 📊 Understanding Results
        - **Normal (🟢)**: Healthy, intact rice grains
        - **Broken (🟠)**: Damaged or broken rice grains
        - **Crack (🔴)**: Rice grains with visible cracks
        
        ### ⚙️ Settings
        - **Confidence Threshold**: Minimum confidence for detections (higher = fewer, more confident detections)
        - **IoU Threshold**: Overlap threshold for removing duplicate detections
        
        ### 💡 Tips
        - Use well-lit, clear images for best results
        - Ensure rice grains are clearly visible
        - Adjust confidence threshold if you're getting too many/few detections
        
        ### 🔧 Command Line Options
        For advanced usage, you can run YOLOv7 directly:
        ```bash
        # Image detection
        cd yolov7
        python detect.py --weights ../models/best.pt --source image.jpg
        
        # Webcam detection
        python detect.py --weights ../models/best.pt --source 0
        
        # Video detection
        python detect.py --weights ../models/best.pt --source video.mp4
        ```
        """)
        
        # System info
        st.subheader("🖥️ System Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **PyTorch Version**: {torch.__version__}
            **Device**: {classifier.device if classifier else 'N/A'}
            **Image Size**: {classifier.img_size if classifier else 'N/A'}
            """)
        
        with col2:
            st.info(f"""
            **Model Classes**: {len(classifier.names) if classifier else 'N/A'}
            **Confidence**: {conf_threshold}
            **IoU Threshold**: {iou_threshold}
            """)

if __name__ == "__main__":
    main()