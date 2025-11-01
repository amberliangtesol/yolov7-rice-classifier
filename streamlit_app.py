#!/usr/bin/env python3
"""
YOLOv7 Rice Quality Classification Streamlit App - Unified Full Version
Supports image upload, video processing, and live classification
Classes: normal, broken, crack
"""

import os
import sys
import streamlit as st
from pathlib import Path
import tempfile
from PIL import Image
import numpy as np
import cv2
import torch
import time
import threading
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# Page configuration
st.set_page_config(
    page_title="🌾 YOLOv7 Rice Quality Classifier",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
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
    .detection-result {
        border: 2px solid #4CAF50;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Global variables
classifier = None

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
        
        # Initialize device
        self.device = self._select_device(device)
        
        # Load model
        self.model = self._load_model()
        
        # Create output directory
        self.output_dir = Path('runs/detect')
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _select_device(self, device=''):
        """Select computation device"""
        if device.lower() == 'cpu':
            return torch.device('cpu')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    def _load_model(self):
        """Load YOLOv7 model with trained weights"""
        try:
            # Add YOLOv7 to path
            yolo_path = Path('./yolov7')
            if yolo_path.exists():
                sys.path.insert(0, str(yolo_path))
            
            # Import YOLOv7 modules
            from models.experimental import attempt_load
            from utils.general import check_img_size, non_max_suppression, scale_coords
            from utils.plots import plot_one_box
            from utils.torch_utils import select_device
            
            # Store functions for later use
            self.attempt_load = attempt_load
            self.check_img_size = check_img_size
            self.non_max_suppression = non_max_suppression
            self.scale_coords = scale_coords
            self.plot_one_box = plot_one_box
            
            # Load model
            model = attempt_load(self.weights_path, map_location=self.device)
            model.eval()
            
            # Check image size
            self.img_size = check_img_size(self.img_size, s=model.stride.max())
            
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
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
    
    def predict_video_frame(self, frame):
        """Run inference on a single video frame"""
        if self.model is None:
            return frame, []
        
        try:
            result_img, detections = self.predict_image(frame)
            return result_img if result_img is not None else frame, detections
        except Exception as e:
            return frame, []
    
    def process_video(self, video_path, output_path=None):
        """Process entire video file"""
        if self.model is None:
            return None, "Model not loaded"
        
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            all_detections = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                result_frame, detections = self.predict_video_frame(frame)
                
                # Store detections with frame info
                for det in detections:
                    det['frame'] = frame_count
                    det['timestamp'] = frame_count / fps
                all_detections.extend(detections)
                
                # Write frame if output specified
                if output_path:
                    out.write(cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR))
                
                frame_count += 1
            
            cap.release()
            if output_path:
                out.release()
            
            return all_detections, f"Processed {frame_count} frames"
            
        except Exception as e:
            return None, f"Error processing video: {e}"
    
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

@st.cache_resource
def load_classifier():
    """Load the rice classifier with caching"""
    weights_path = 'models/best.pt'
    if not os.path.exists(weights_path):
        return None, f"Model file not found at {weights_path}"
    
    try:
        classifier = RiceClassifierStreamlit(weights_path=weights_path)
        if classifier.model is None:
            return None, "Failed to load model"
        return classifier, "Model loaded successfully!"
    except Exception as e:
        return None, f"Error initializing classifier: {e}"

def predict_image_interface(image, conf_threshold, iou_threshold):
    """Main prediction interface"""
    global classifier
    
    if classifier is None:
        classifier_obj, status = load_classifier()
        if classifier_obj is None:
            return None, status
        classifier = classifier_obj
    
    # Update thresholds
    classifier.conf_thres = conf_threshold
    classifier.iou_thres = iou_threshold
    
    # Run prediction
    result_img, detections = classifier.predict_image(image)
    
    if result_img is None:
        return None, detections
    
    return result_img, detections

def create_detection_summary(detections):
    """Create a summary of detections"""
    if not detections:
        return "No rice grains detected. Try adjusting the confidence threshold."
    
    # Count by class
    class_counts = {'normal': 0, 'broken': 0, 'crack': 0}
    for det in detections:
        class_counts[det['class']] += 1
    
    total = len(detections)
    summary = f"**Total detected: {total} rice grains**\n\n"
    
    # Add percentages
    for class_name, count in class_counts.items():
        percentage = (count / total * 100) if total > 0 else 0
        emoji = {'normal': '🟢', 'broken': '🟠', 'crack': '🔴'}[class_name]
        summary += f"{emoji} **{class_name.capitalize()}**: {count} ({percentage:.1f}%)\n"
    
    return summary

class VideoTransformer(VideoProcessorBase):
    """Video transformer for webcam processing"""
    
    def __init__(self):
        self.classifier = None
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
    
    def set_classifier(self, classifier, conf_threshold, iou_threshold):
        self.classifier = classifier
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if self.classifier is not None:
            # Update thresholds
            self.classifier.conf_thres = self.conf_threshold
            self.classifier.iou_thres = self.iou_threshold
            
            # Process frame
            result_img, detections = self.classifier.predict_video_frame(img)
            
            # Convert back to BGR for video output
            if result_img is not None:
                result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
                return av.VideoFrame.from_ndarray(result_bgr, format="bgr24")
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def process_video_interface(video_file, conf_threshold, iou_threshold):
    """Video processing interface"""
    global classifier
    
    if classifier is None:
        classifier_obj, status = load_classifier()
        if classifier_obj is None:
            return None, status
        classifier = classifier_obj
    
    # Update thresholds
    classifier.conf_thres = conf_threshold
    classifier.iou_thres = iou_threshold
    
    # Save uploaded video to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_file.read())
        temp_video_path = tmp_file.name
    
    # Process video
    detections, status = classifier.process_video(temp_video_path)
    
    # Clean up temp file
    os.unlink(temp_video_path)
    
    return detections, status

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🌾 YOLOv7 Rice Quality Classifier</h1>', unsafe_allow_html=True)
    
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
        
        # System info
        st.subheader("🖥️ System Info")
        try:
            device = "CUDA" if torch.cuda.is_available() else "CPU"
            st.text(f"Device: {device}")
            st.text(f"PyTorch: {torch.__version__}")
        except:
            st.text("System info unavailable")
    
    # Check if model can be loaded
    classifier_obj, status = load_classifier()
    
    if classifier_obj is None:
        st.error(f"❌ {status}")
        st.info("""
        **Model Loading Issue**: The YOLOv7 model couldn't be loaded. This might be due to:
        - Missing model file (best.pt)
        - Missing dependencies
        - Cloud environment limitations
        
        **What you can do**:
        1. Check that `models/best.pt` exists
        2. Verify all dependencies are installed
        3. Try refreshing the page
        """)
        return
    else:
        st.success(f"✅ {status}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📷 Image Upload", "📹 Video Processing", "📸 Live Camera", "📊 Batch Analysis"])
    
    with tab1:
        st.header("📷 Image Upload Classification")
        
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
                
                # Image info
                st.text(f"Size: {image.size[0]} x {image.size[1]}")
                st.text(f"Format: {image.format}")
            
            with col2:
                st.subheader("Detection Results")
                
                # Run prediction
                with st.spinner("🔍 Analyzing image..."):
                    result_img, detections = predict_image_interface(
                        image, conf_threshold, iou_threshold
                    )
                
                if result_img is not None:
                    st.image(result_img, caption="Detection Results", use_column_width=True)
                    
                    # Display detection summary
                    summary = create_detection_summary(detections)
                    st.markdown(summary)
                    
                    # Detailed results
                    if detections:
                        with st.expander(f"📋 Detailed Results ({len(detections)} detections)"):
                            for i, det in enumerate(detections):
                                emoji = {'normal': '🟢', 'broken': '🟠', 'crack': '🔴'}[det['class']]
                                st.write(f"{emoji} **Detection {i+1}**: {det['class']} (confidence: {det['confidence']:.3f})")
                                st.write(f"   📍 Bounding box: ({det['bbox'][0]}, {det['bbox'][1]}) to ({det['bbox'][2]}, {det['bbox'][3]})")
                    
                    # Download results
                    if st.button("💾 Save Results"):
                        # Convert image to bytes for download
                        import io
                        img_bytes = io.BytesIO()
                        Image.fromarray(result_img).save(img_bytes, format='PNG')
                        st.download_button(
                            label="📥 Download Detection Image",
                            data=img_bytes.getvalue(),
                            file_name="rice_detection_results.png",
                            mime="image/png"
                        )
                else:
                    st.error(f"❌ Prediction failed: {detections}")
    
    with tab2:
        st.header("📹 Video Processing")
        
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov'],
            help="Upload a video file for rice grain detection"
        )
        
        if uploaded_video is not None:
            st.video(uploaded_video)
            
            if st.button("🎬 Process Video"):
                with st.spinner("🔄 Processing video..."):
                    detections, status = process_video_interface(
                        uploaded_video, conf_threshold, iou_threshold
                    )
                
                if detections is not None:
                    st.success(f"✅ {status}")
                    
                    # Video analysis summary
                    if detections:
                        st.subheader("📊 Video Analysis Results")
                        
                        # Count detections by class
                        class_counts = {'normal': 0, 'broken': 0, 'crack': 0}
                        frame_count = 0
                        if detections:
                            frame_count = max([d.get('frame', 0) for d in detections]) + 1
                            for det in detections:
                                class_counts[det['class']] += 1
                        
                        # Display metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Frames", frame_count)
                        with col2:
                            st.metric("🟢 Normal", class_counts['normal'])
                        with col3:
                            st.metric("🟠 Broken", class_counts['broken'])
                        with col4:
                            st.metric("🔴 Crack", class_counts['crack'])
                        
                        # Detection timeline
                        if len(detections) > 0:
                            with st.expander("📈 Detection Timeline"):
                                import pandas as pd
                                df = pd.DataFrame(detections)
                                if 'timestamp' in df.columns:
                                    st.line_chart(df.groupby(['timestamp', 'class']).size().unstack(fill_value=0))
                    else:
                        st.info("No rice grains detected in the video.")
                else:
                    st.error(f"❌ Video processing failed: {status}")
    
    with tab3:
        st.header("📸 Live Camera")
        
        # Check if model can be loaded
        classifier_obj, status = load_classifier()
        
        if classifier_obj is not None:
            st.info("🎥 Live camera detection with YOLOv7")
            
            # Camera settings
            st.subheader("⚙️ Camera Settings")
            col1, col2 = st.columns(2)
            with col1:
                camera_conf = st.slider("Camera Confidence", 0.1, 1.0, conf_threshold, 0.05, key="camera_conf")
            with col2:
                camera_iou = st.slider("Camera IoU", 0.1, 1.0, iou_threshold, 0.05, key="camera_iou")
            
            # WebRTC streamer  
            video_transformer = VideoTransformer()
            if classifier_obj is not None:
                video_transformer.set_classifier(classifier_obj, camera_conf, camera_iou)
            
            webrtc_ctx = webrtc_streamer(
                key="rice-detection",
                video_processor_factory=lambda: video_transformer,
                rtc_configuration=RTCConfiguration(
                    ice_servers=[
                        # Google STUN servers
                        {"urls": ["stun:stun.l.google.com:19302"]},
                        {"urls": ["stun:stun1.l.google.com:19302"]},
                        {"urls": ["stun:stun2.l.google.com:19302"]},
                        {"urls": ["stun:stun3.l.google.com:19302"]},
                        {"urls": ["stun:stun4.l.google.com:19302"]},
                        # Alternative STUN servers
                        {"urls": ["stun:stun.services.mozilla.com"]},
                        {"urls": ["stun:stun.stunprotocol.org:3478"]},
                        # Public TURN servers (for difficult networks)
                        {
                            "urls": ["turn:openrelay.metered.ca:80"],
                            "username": "openrelayproject",
                            "credential": "openrelayproject"
                        },
                        {
                            "urls": ["turn:openrelay.metered.ca:443"],
                            "username": "openrelayproject", 
                            "credential": "openrelayproject"
                        }
                    ],
                    ice_candidate_pool_size=10
                ),
                media_stream_constraints={
                    "video": {
                        "width": {"min": 640, "ideal": 1280, "max": 1920},
                        "height": {"min": 480, "ideal": 720, "max": 1080},
                        "frameRate": {"min": 10, "ideal": 15, "max": 30}
                    },
                    "audio": False
                },
                async_processing=True,
            )
            
            # Connection status
            if webrtc_ctx.state.playing:
                st.success("🟢 Camera connected successfully!")
            elif webrtc_ctx.state.signalling:
                st.info("🔄 Connecting to camera...")
            else:
                st.warning("⚠️ Camera not connected. Click START to begin.")
            
            st.markdown("""
            **📱 How to use Live Camera:**
            1. Click "START" to begin camera detection
            2. Allow browser camera access when prompted
            3. Position rice grains in front of camera
            4. Adjust confidence/IoU thresholds as needed
            5. Click "STOP" when finished
            
            **🔧 Connection Troubleshooting:**
            - Use **Chrome** or **Firefox** browser
            - Ensure camera permission is granted
            - Try refreshing the page if connection fails
            - Check if other apps are using the camera
            - On mobile: try switching between front/back camera
            """)
            
            # Add connection help
            with st.expander("🛠️ Advanced Connection Settings"):
                st.markdown("""
                **Current STUN/TURN Servers:**
                - Google STUN: stun.l.google.com:19302 (+ 4 backup)
                - Mozilla STUN: stun.services.mozilla.com
                - Standard STUN: stun.stunprotocol.org:3478
                - Public TURN: openrelay.metered.ca (for difficult networks)
                
                **Video Quality Settings:**
                - Resolution: 640x480 to 1920x1080
                - Frame Rate: 10-30 FPS
                - Optimized for rice grain detection
                
                **If connection still fails:**
                1. Try using the "📷 Image Upload" tab instead
                2. Take photos with your phone and upload them
                3. Use the "📹 Video Processing" for recorded videos
                """)
        else:
            st.error(f"❌ Model loading failed: {status}")
        
        # Simple camera alternative
        st.markdown("---")
        st.subheader("📱 Simple Camera Alternative")
        st.info("💡 **If WebRTC doesn't work**: Use Streamlit's built-in camera!")
        
        simple_camera = st.camera_input("📸 Take a photo")
        if simple_camera is not None:
            # Process the captured image
            image = Image.open(simple_camera)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="📱 Captured Image", use_column_width=True)
            
            with col2:
                with st.spinner("🔍 Analyzing image..."):
                    result_img, detections = predict_image_interface(
                        image, conf_threshold, iou_threshold
                    )
                
                if result_img is not None:
                    st.image(result_img, caption="🎯 Detection Results", use_column_width=True)
                    summary = create_detection_summary(detections)
                    st.markdown(summary)
                else:
                    st.error(f"❌ Analysis failed: {detections}")
    
    with tab4:
        st.header("📊 Batch Analysis")
        st.info("🚧 Batch processing feature coming soon!")
        
        uploaded_files = st.file_uploader(
            "Choose multiple image files",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload multiple images for batch analysis"
        )
        
        if uploaded_files:
            st.write(f"📁 Uploaded {len(uploaded_files)} images")
            if st.button("🔄 Process All Images"):
                progress_bar = st.progress(0)
                results = []
                
                for i, file in enumerate(uploaded_files):
                    image = Image.open(file)
                    result_img, detections = predict_image_interface(
                        image, conf_threshold, iou_threshold
                    )
                    
                    if result_img is not None:
                        results.append({
                            'filename': file.name,
                            'detections': len(detections),
                            'normal': len([d for d in detections if d['class'] == 'normal']),
                            'broken': len([d for d in detections if d['class'] == 'broken']),
                            'crack': len([d for d in detections if d['class'] == 'crack'])
                        })
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Display batch results
                if results:
                    import pandas as pd
                    df = pd.DataFrame(results)
                    st.subheader("📈 Batch Analysis Results")
                    st.dataframe(df)
                    
                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Images", len(results))
                    with col2:
                        st.metric("🟢 Normal", df['normal'].sum())
                    with col3:
                        st.metric("🟠 Broken", df['broken'].sum())
                    with col4:
                        st.metric("🔴 Crack", df['crack'].sum())

if __name__ == "__main__":
    main()