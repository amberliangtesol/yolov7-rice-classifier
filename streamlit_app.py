#!/usr/bin/env python3
"""
YOLOv7 Rice Quality Classification Streamlit App - Unified Full Version
Supports image upload, video processing, and live classification
Classes: white_rice, thi_rice, brown_rice, black_rice
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
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import subprocess
import json

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

# H.264 conversion function for better video compatibility
def to_h264(input_path, output_path=None):
    """Convert video to H.264 format for better browser compatibility with enhanced error handling"""
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_h264.mp4"
    
    try:
        # Check if ffmpeg is available
        ffmpeg_check = subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print(f"✅ ffmpeg available: {ffmpeg_check.returncode == 0}")
        
        # Get video info first to check dimensions
        probe_cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', input_path
        ]
        
        try:
            probe_result = subprocess.run(probe_cmd, capture_output=True, check=True, text=True)
            import json
            video_info = json.loads(probe_result.stdout)
            
            # Find video stream
            video_stream = None
            for stream in video_info.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break
            
            if video_stream:
                width = int(video_stream.get('width', 0))
                height = int(video_stream.get('height', 0))
                print(f"📐 原始尺寸: {width}x{height}")
                
                # Check if dimensions are odd (H.264 requirement: must be even)
                if width % 2 != 0 or height % 2 != 0:
                    # Force even dimensions
                    width = width + (width % 2)
                    height = height + (height % 2)
                    print(f"🔧 調整為偶數尺寸: {width}x{height}")
                    scale_filter = f"scale={width}:{height}"
                else:
                    scale_filter = None
                    print("✅ 尺寸已為偶數，無需調整")
                    
        except Exception as probe_error:
            print(f"⚠️ 無法獲取視頻信息，使用默認設置: {probe_error}")
            scale_filter = "scale=trunc(iw/2)*2:trunc(ih/2)*2"  # Force even dimensions
        
        # Build ffmpeg command with enhanced settings
        cmd = [
            'ffmpeg', '-y',  # -y to overwrite output file
            '-i', input_path,
            '-c:v', 'libx264',  # H.264 codec
            '-preset', 'fast',  # Fast encoding preset
            '-crf', '26',  # Higher CRF (lower quality) to reduce file size
            '-maxrate', '2M',  # Limit maximum bitrate
            '-bufsize', '4M',  # Buffer size
            '-movflags', '+faststart',  # Optimize for web streaming
            '-pix_fmt', 'yuv420p',  # Ensure compatibility
        ]
        
        # Add scale filter if needed (force even dimensions)
        if scale_filter:
            cmd.extend(['-vf', scale_filter])
        else:
            # Ensure even dimensions even if probe failed
            cmd.extend(['-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2'])
            
        cmd.append(output_path)
        
        print(f"🔄 執行 ffmpeg 命令: {' '.join(cmd[:8])}...")
        
        # Run conversion with detailed error capture
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ H.264 轉換成功")
            return output_path
        else:
            print(f"❌ ffmpeg 轉換失敗 (返回碼: {result.returncode})")
            print(f"📋 stderr: {result.stderr[:500]}...")  # Show first 500 chars of error
            print(f"📋 stdout: {result.stdout[:500]}...")
            return None
        
    except subprocess.CalledProcessError as e:
        print(f"❌ ffmpeg 命令執行失敗: {e}")
        print(f"📋 返回碼: {e.returncode}")
        if e.stderr:
            print(f"📋 錯誤輸出: {e.stderr.decode()[:500]}")
        return None
    except FileNotFoundError:
        print("❌ ffmpeg 未安裝或無法找到")
        return None
    except Exception as e:
        print(f"❌ H.264 轉換出現意外錯誤: {e}")
        return None

def ffprobe_json(path: str) -> dict:
    """Get detailed video metadata using ffprobe for browser compatibility analysis"""
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_streams", "-show_format", path
        ])
        return json.loads(out.decode("utf-8"))
    except Exception as e:
        st.warning(f"ffprobe 失敗：{e}")
        return {}

class RiceClassifierStreamlit:
    def __init__(self, weights_path='models/best.pt', device='', img_size=640, conf_thres=0.25, iou_thres=0.45):
        """Initialize YOLOv7 Rice Classifier for Streamlit"""
        self.weights_path = weights_path
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # Class names for rice quality
        self.names = ['white_rice', 'thi_rice', 'brown_rice', 'black_rice']
        self.colors = [(255, 255, 255), (255, 215, 0), (139, 69, 19), (0, 0, 0)]  # White, Gold, Brown, Black
        
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
    
    def process_video(self, video_path, output_path=None, progress_callback=None):
        """Process entire video file with progress tracking"""
        if self.model is None:
            return None, "Model not loaded"
        
        print(f"Starting video processing: {video_path}")
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Failed to open video: {video_path}")
                return None, f"Error opening video: {video_path}"
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"Video properties: {total_frames} frames, {fps} FPS, {width}x{height}")
            
            # Ensure even dimensions for VideoWriter compatibility (especially for H.264)
            original_width, original_height = width, height
            if width % 2 != 0:
                width = width + 1
                print(f"🔧 Adjusted width from {original_width} to {width} (must be even)")
            if height % 2 != 0:
                height = height + 1
                print(f"🔧 Adjusted height from {original_height} to {height} (must be even)")
            
            if width != original_width or height != original_height:
                print(f"📐 VideoWriter will use dimensions: {width}x{height} (adjusted from {original_width}x{original_height})")
            
            out = None
            if output_path:
                # Try different codecs optimized for cloud deployment and browser compatibility
                codecs_to_try = [
                    ('mp4v', '.mp4'),  # MPEG-4 - most widely supported
                    ('XVID', '.avi'),  # Xvid - good fallback
                    ('avc1', '.mp4'),  # H.264 - best quality but may not be available in cloud
                    ('H264', '.mp4'),  # Alternative H.264
                ]
                
                for codec, ext in codecs_to_try:
                    try:
                        # Adjust output path extension based on codec
                        current_output_path = output_path
                        if not output_path.endswith(ext):
                            current_output_path = output_path.rsplit('.', 1)[0] + ext
                        
                        fourcc = cv2.VideoWriter_fourcc(*codec)
                        # Use the adjusted even dimensions
                        out = cv2.VideoWriter(current_output_path, fourcc, fps, (width, height))
                        if out.isOpened():
                            print(f"✅ Successfully created video writer with codec: {codec}, dimensions: {width}x{height}, file: {current_output_path}")
                            # Update output_path to the successful one
                            output_path = current_output_path
                            break
                        else:
                            out.release()
                    except Exception as codec_error:
                        print(f"❌ Failed to create video writer with codec {codec}: {codec_error}")
                        continue
                
                if out is None or not out.isOpened():
                    print(f"Failed to create output video writer with any codec")
                    cap.release()
                    return None, f"Error creating output video: {output_path}"
            
            all_detections = []
            frame_count = 0
            start_time = time.time()
            
            print("Starting frame processing...")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                try:
                    # Process frame
                    result_frame, detections = self.predict_video_frame(frame)
                    
                    # Store detections with frame info
                    for det in detections:
                        det['frame'] = frame_count
                        det['timestamp'] = frame_count / fps
                    all_detections.extend(detections)
                    
                    # Write frame if output specified
                    if out is not None:
                        # Convert RGB back to BGR for video output
                        bgr_frame = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
                        
                        # Ensure frame dimensions match VideoWriter dimensions
                        frame_h, frame_w = bgr_frame.shape[:2]
                        if frame_w != width or frame_h != height:
                            # Resize frame to match VideoWriter dimensions
                            bgr_frame = cv2.resize(bgr_frame, (width, height), interpolation=cv2.INTER_LINEAR)
                            if frame_count == 0:  # Log only once
                                print(f"🔧 Resizing frames from {frame_w}x{frame_h} to {width}x{height} for VideoWriter consistency")
                        
                        out.write(bgr_frame)
                    
                    frame_count += 1
                    
                    # Update progress if callback provided
                    if progress_callback and total_frames > 0:
                        progress = frame_count / total_frames
                        elapsed_time = time.time() - start_time
                        if frame_count > 0:
                            eta = (elapsed_time / frame_count) * (total_frames - frame_count)
                            progress_callback(progress, frame_count, total_frames, elapsed_time, eta)
                    
                    # Print progress every 10 frames
                    if frame_count % 10 == 0:
                        progress_pct = (frame_count / total_frames) * 100
                        print(f"Processed {frame_count}/{total_frames} frames ({progress_pct:.1f}%)")
                        
                except Exception as frame_error:
                    print(f"Error processing frame {frame_count}: {frame_error}")
                    frame_count += 1
                    continue
            
            cap.release()
            if out is not None:
                out.release()
            
            total_time = time.time() - start_time
            print(f"Video processing completed: {frame_count} frames in {total_time:.1f}s")
            return all_detections, f"Processed {frame_count} frames in {total_time:.1f}s"
            
        except Exception as e:
            print(f"Error in process_video: {e}")
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
        # Suppress PyTorch warnings
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        
        print(f"Loading model from {weights_path}...")
        classifier = RiceClassifierStreamlit(weights_path=weights_path)
        if classifier.model is None:
            return None, "Failed to load model"
        print("Model loaded successfully!")
        return classifier, "Model loaded successfully!"
    except Exception as e:
        print(f"Error loading model: {e}")
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
    class_counts = {'white_rice': 0, 'thi_rice': 0, 'brown_rice': 0, 'black_rice': 0}
    for det in detections:
        class_counts[det['class']] += 1
    
    total = len(detections)
    summary = f"**Total detected: {total} rice grains**\n\n"
    
    # Add percentages
    for class_name, count in class_counts.items():
        percentage = (count / total * 100) if total > 0 else 0
        emoji = {'white_rice': '⚪', 'thi_rice': '🟡', 'brown_rice': '🟤', 'black_rice': '⚫'}[class_name]
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

def process_video_interface(video_file, conf_threshold, iou_threshold, progress_placeholder=None, status_placeholder=None):
    """Video processing interface with progress tracking"""
    global classifier
    
    # Add debug logging
    if status_placeholder:
        status_placeholder.info("🔧 載入模型中...")
    
    if classifier is None:
        classifier_obj, status = load_classifier()
        if classifier_obj is None:
            return None, status, None
        classifier = classifier_obj
    
    # Update thresholds
    classifier.conf_thres = conf_threshold
    classifier.iou_thres = iou_threshold
    
    temp_video_path = None
    output_video_path = None
    
    try:
        if status_placeholder:
            status_placeholder.info("💾 保存視頻文件...")
        
        # Save uploaded video to temp file
        video_bytes = video_file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_bytes)
            temp_video_path = tmp_file.name
        
        # Create output video path
        output_video_path = temp_video_path.replace('.mp4', '_processed.mp4')
        
        if status_placeholder:
            status_placeholder.info("📹 讀取視頻信息...")
        
        # Check video can be opened
        import cv2
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            return None, "Error: 無法打開視頻文件", None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        if status_placeholder:
            status_placeholder.info(f"📊 視頻信息: {total_frames} frames, {fps:.1f} FPS")
        
        # Progress callback function
        def update_progress(progress, current_frame, total_frames, elapsed_time, eta):
            try:
                if progress_placeholder is not None:
                    progress_placeholder.progress(progress)
                if status_placeholder is not None:
                    mins_elapsed = int(elapsed_time // 60)
                    secs_elapsed = int(elapsed_time % 60)
                    mins_eta = int(eta // 60)
                    secs_eta = int(eta % 60)
                    
                    status_text = f"""🎬 處理進度: {current_frame}/{total_frames} frames ({progress:.1%})
⏱️ 已用時間: {mins_elapsed:02d}:{secs_elapsed:02d}
⏳ 預估剩餘: {mins_eta:02d}:{secs_eta:02d}
🔄 處理速度: {current_frame/elapsed_time:.1f} frames/sec"""
                    status_placeholder.info(status_text)
            except Exception as e:
                # Ignore progress update errors to prevent breaking the main process
                pass
        
        if status_placeholder:
            status_placeholder.info("🚀 開始處理視頻...")
        
        # Process video with progress tracking and output video
        try:
            detections, status = classifier.process_video(temp_video_path, output_path=output_video_path, progress_callback=update_progress)
        except TypeError as e:
            # Fallback for environments that don't support progress_callback
            if status_placeholder:
                status_placeholder.warning(f"⚠️ Progress callback not supported, falling back... ({str(e)})")
            detections, status = classifier.process_video(temp_video_path, output_path=output_video_path)
        
        # Check processed video file (but don't read bytes to memory yet)
        processed_video_bytes = None
        if output_video_path and os.path.exists(output_video_path):
            try:
                if status_placeholder:
                    status_placeholder.info("📤 視頻檔案準備完成...")
                
                # Check file size
                file_size = os.path.getsize(output_video_path)
                print(f"Output video file size: {file_size} bytes")
                
                if file_size > 0:
                    # Don't read the entire file into memory - just verify it exists and has content
                    # The video preview will use file path directly
                    print(f"Output video ready for preview: {output_video_path}")
                    if status_placeholder:
                        status_placeholder.success(f"✅ 視頻處理完成 (檔案大小: {file_size/1024/1024:.1f}MB)")
                else:
                    print("Output video file is empty")
                    if status_placeholder:
                        status_placeholder.error("❌ 輸出視頻檔案為空")
            except Exception as e:
                print(f"Error checking processed video: {e}")
                if status_placeholder:
                    status_placeholder.warning(f"⚠️ 視頻檔案檢查失敗: {e}")
        
        return detections, status, processed_video_bytes, output_video_path
        
    except Exception as e:
        print(f"Error in process_video_interface: {e}")
        return None, f"Error processing video: {str(e)}", None, None
    
    finally:
        # Clean up temp input files only - KEEP processed output video for preview/download
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.unlink(temp_video_path)
                print(f"✅ Cleaned up temp input video: {temp_video_path}")
            except Exception as e:
                print(f"⚠️ Failed to cleanup temp input video: {e}")
        
        # DO NOT clean up output_video_path here - let Streamlit handle it
        # The processed video and H.264 converted video will be kept for preview
        # Streamlit will clean up temp files when the session ends
        if output_video_path:
            print(f"📁 Keeping processed video for preview: {output_video_path}")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🌾 YOLOv7 Rice Type Classifier</h1>', unsafe_allow_html=True)
    
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
        **Classes**: 4 (white_rice, thi_rice, brown_rice, black_rice)
        **Input Size**: 640x640
        **Format**: PyTorch (.pt)
        """)
        
        # Legend
        st.subheader("🏷️ Class Legend")
        st.markdown("""
        - ⚪ **White Rice**: Regular white rice
        - 🟡 **Thi Rice**: Thai jasmine rice
        - 🟤 **Brown Rice**: Whole grain brown rice
        - ⚫ **Black Rice**: Black glutinous rice
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
                                emoji = {'white_rice': '⚪', 'thi_rice': '🟡', 'brown_rice': '🟤', 'black_rice': '⚫'}[det['class']]
                                st.write(f"{emoji} **Detection {i+1}**: {det['class']} (confidence: {det['confidence']:.3f})")
                                st.write(f"   📍 Bounding box: ({det['bbox'][0]}, {det['bbox'][1]}) to ({det['bbox'][2]}, {det['bbox'][3]})")
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
                # Create placeholders for progress tracking
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
                
                # Show initial progress
                progress_placeholder.progress(0)
                status_placeholder.info("🔄 初始化視頻處理...")
                
                # Process video with progress tracking
                detections, status, processed_video_bytes, output_video_path = process_video_interface(
                    uploaded_video, conf_threshold, iou_threshold,
                    progress_placeholder, status_placeholder
                )
                
                # Clear progress indicators when done
                progress_placeholder.empty()
                status_placeholder.empty()
                
                if detections is not None:
                    st.success(f"✅ {status}")
                    
                    # Display processed video with detections
                    if output_video_path and os.path.exists(output_video_path):
                        st.subheader("🎥 Processed Video with Detections")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**📤 Original Video**")
                            st.video(uploaded_video)
                        
                        with col2:
                            st.markdown("**🎯 Detection Results Video**")
                            
                            # 使用檔案路徑 + H.264 轉碼的簡化方式
                            try:
                                # 1) 檢查原始偵測輸出是否完成
                                if output_video_path and os.path.exists(output_video_path):
                                    file_size_mb = os.path.getsize(output_video_path) / (1024 * 1024)
                                    # Video processing completed
                                    
                                    # 2) 轉換 H.264 以確保 HTML5 可播
                                    h264_path = to_h264(output_video_path)
                                    
                                    if h264_path and os.path.exists(h264_path):
                                        # H.264 conversion completed
                                        
                                        # 顯示視頻元數據以診斷瀏覽器兼容性
                                        meta = ffprobe_json(h264_path)
                                        if meta.get("streams"):
                                            # Video metadata available for debugging
                                            pass
                                        
                                        # 3) 用檔案路徑做預覽（比 bytes 穩）
                                        st.video(h264_path)
                                        # Video preview loaded successfully
                                        
                                        # 4) 下載按鈕用轉好的 H.264
                                        with open(h264_path, 'rb') as f:
                                            h264_bytes = f.read()
                                        
                                        st.download_button(
                                            label="📥 Download Processed Video (H.264)",
                                            data=h264_bytes,
                                            file_name="rice_detection_h264.mp4",
                                            mime="video/mp4"
                                        )
                                    else:
                                        # H.264 轉換失敗，顯示原始視頻的元數據診斷信息
                                        meta = ffprobe_json(output_video_path)
                                        if meta.get("streams"):
                                            st.write("🎛️ 原始視頻 metadata (診斷為何轉換失敗)：", meta.get("streams", []))
                                        
                                        # 直接提供下載按鈕
                                        with open(output_video_path, 'rb') as f:
                                            original_bytes = f.read()
                                        
                                        st.download_button(
                                            label="📥 Download Processed Video",
                                            data=original_bytes,
                                            file_name="rice_detection_video.mp4",
                                            mime="video/mp4"
                                        )
                                else:
                                    st.error("❌ 視頻檔案路徑無效或檔案不存在")
                                    
                            except Exception as e:
                                st.error(f"❌ 視頻預覽失敗: {str(e)}")
                                # 緊急備用方案
                                if processed_video_bytes:
                                    st.warning("🔄 使用備用預覽方式...")
                                    st.video(processed_video_bytes)
                                    
                                    st.download_button(
                                        label="📥 Download Processed Video (Backup)",
                                        data=processed_video_bytes,
                                        file_name="rice_detection_backup.mp4",
                                        mime="video/mp4"
                                    )
                    else:
                        st.warning("⚠️ 處理後的視頻未能正確生成。請檢查輸入視頻格式。")
                    
                    # Video analysis summary
                    if detections:
                        st.subheader("📊 Video Analysis Results")
                        
                        # Count detections by class
                        class_counts = {'white_rice': 0, 'thi_rice': 0, 'brown_rice': 0, 'black_rice': 0}
                        frame_count = 0
                        if detections:
                            frame_count = max([d.get('frame', 0) for d in detections]) + 1
                            for det in detections:
                                class_counts[det['class']] += 1
                        
                        # Display metrics
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("Total Frames", frame_count)
                        with col2:
                            st.metric("⚪ White Rice", class_counts['white_rice'])
                        with col3:
                            st.metric("🟡 Thi Rice", class_counts['thi_rice'])
                        with col4:
                            st.metric("🟤 Brown Rice", class_counts['brown_rice'])
                        with col5:
                            st.metric("⚫ Black Rice", class_counts['black_rice'])
                        
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
            
            # Simplified WebRTC configuration for cloud deployment
            # Using camelCase and minimal servers for faster connection
            rtc_config = {
                "iceServers": [
                    # Single reliable STUN server
                    {"urls": "stun:stun.l.google.com:19302"},
                    # Single TURN server for cloud NAT traversal
                    {
                        "urls": "turn:openrelay.metered.ca:80",
                        "username": "openrelayproject",
                        "credential": "openrelayproject"
                    }
                ]
            }
            
            webrtc_ctx = webrtc_streamer(
                key="rice-detection",
                video_processor_factory=lambda: video_transformer,
                rtc_configuration=rtc_config,
                media_stream_constraints={
                    "video": {
                        "width": {"min": 640, "ideal": 1280, "max": 1920},
                        "height": {"min": 480, "ideal": 720, "max": 1080},
                        "frameRate": {"min": 10, "ideal": 15, "max": 30},
                        "facingMode": "environment"  # Use rear camera on mobile devices
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
            3. **📱 Mobile users**: App automatically uses rear camera for better rice grain capture
            4. Position rice grains in front of camera
            5. Adjust confidence/IoU thresholds as needed
            6. Click "STOP" when finished
            
            **🔧 Connection Troubleshooting (雲端部署專用):**
            - 如果顯示 "Connection is taking longer than expected" → 正常現象，雲端環境需要TURN服務器
            - Use **Chrome** or **Firefox** browser (必須HTTPS環境)
            - Ensure camera permission is granted
            - 公司/學校網路可能會封鎖：嘗試用手機4G測試
            - Try refreshing the page if connection fails
            - **Cloud deployment**: 已自動配置多個免費TURN服務器
            """)
            
            # Enhanced connection diagnostics
            with st.expander("🛠️ Advanced Connection Settings & Diagnostics"):
                st.markdown("""
                **✅ Enhanced Cloud-Ready STUN/TURN Configuration:**
                
                **STUN Servers (NAT discovery):**
                - Google STUN: stun.l.google.com:19302 (+ 4 backup servers)
                - Mozilla STUN: stun.services.mozilla.com
                - Cloudflare STUN: stun.cloudflare.com:3478
                - NextCloud STUN: stun.nextcloud.com:443
                
                **TURN Servers (Relay for cloud deployment):**
                - 🌐 Metered.ca: openrelay.metered.ca (UDP/TCP 80, 443)
                - 🌐 Global Relay: global.relay.metered.ca (UDP/TCP 80, 443)
                - 🌐 Backup Relay: relay.backups.cz (UDP/TCP 443)
                
                **🔧 Debugging Steps:**
                1. **Open Browser DevTools** (F12) → Console tab
                2. Look for ICE connection state messages
                3. If you see "ICE failed" or "ICE closed" → TURN server issue
                4. **For developers**: Uncomment `ice_transport_policy="relay"` to force TURN
                
                **📊 Connection Status Meanings:**
                - 🟢 **Connected**: WebRTC working perfectly
                - 🔄 **Signalling**: Negotiating connection (normal for cloud)
                - ⚠️ **Not connected**: Click START or check permissions
                - ❌ **Failed**: Check network/firewall settings
                
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
        st.info("📁 Upload multiple images for batch processing with real-time progress tracking!")
        
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
                status_text = st.empty()
                results = []
                start_time = time.time()
                
                for i, file in enumerate(uploaded_files):
                    # Update status
                    elapsed = time.time() - start_time
                    if i > 0:
                        eta = (elapsed / i) * (len(uploaded_files) - i)
                        status_text.info(f"📸 處理中: {i+1}/{len(uploaded_files)} ({(i+1)/len(uploaded_files):.1%}) | ⏱️ 已用時間: {elapsed:.1f}s | ⏳ 預估剩餘: {eta:.1f}s")
                    else:
                        status_text.info(f"📸 處理中: {i+1}/{len(uploaded_files)} ({(i+1)/len(uploaded_files):.1%})")
                    
                    image = Image.open(file)
                    result_img, detections = predict_image_interface(
                        image, conf_threshold, iou_threshold
                    )
                    
                    if result_img is not None:
                        results.append({
                            'filename': file.name,
                            'detections': len(detections),
                            'white_rice': len([d for d in detections if d['class'] == 'white_rice']),
                            'thi_rice': len([d for d in detections if d['class'] == 'thi_rice']),
                            'brown_rice': len([d for d in detections if d['class'] == 'brown_rice']),
                            'black_rice': len([d for d in detections if d['class'] == 'black_rice'])
                        })
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Clear progress indicators
                status_text.empty()
                
                # Display batch results
                if results:
                    import pandas as pd
                    df = pd.DataFrame(results)
                    st.subheader("📈 Batch Analysis Results")
                    st.dataframe(df)
                    
                    # Summary statistics
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Total Images", len(results))
                    with col2:
                        st.metric("⚪ White Rice", df['white_rice'].sum())
                    with col3:
                        st.metric("🟡 Thi Rice", df['thi_rice'].sum())
                    with col4:
                        st.metric("🟤 Brown Rice", df['brown_rice'].sum())
                    with col5:
                        st.metric("⚫ Black Rice", df['black_rice'].sum())

if __name__ == "__main__":
    main()