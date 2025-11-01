#!/usr/bin/env python3
"""
YOLOv7 Rice Quality Classification Web Demo
Supports image upload, webcam capture, and video processing
Classes: normal, broken, crack
"""

import os
import sys
import argparse
import time
import cv2
import torch
import numpy as np
import gradio as gr
from pathlib import Path
import tempfile
import shutil

# Add YOLOv7 to path
yolo_path = Path('./yolov7')
if yolo_path.exists():
    sys.path.append(str(yolo_path))

try:
    from models.experimental import attempt_load
    from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
    from utils.plots import plot_one_box
    from utils.torch_utils import select_device, load_classifier, time_synchronized
    from utils.datasets import LoadImages, LoadStreams
except ImportError:
    print("YOLOv7 modules not found. Please run setup script first.")
    sys.exit(1)

class RiceClassifier:
    def __init__(self, weights_path='models/best.pt', device='', img_size=640, conf_thres=0.25, iou_thres=0.45):
        """
        Initialize YOLOv7 Rice Classifier
        
        Args:
            weights_path: Path to trained model weights
            device: Device to use ('cpu', 'cuda', or auto-detect)
            img_size: Input image size
            conf_thres: Confidence threshold
            iou_thres: IoU threshold for NMS
        """
        self.weights_path = weights_path
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # Class names for rice quality
        self.names = ['normal', 'broken', 'crack']
        self.colors = [(0, 255, 0), (255, 165, 0), (255, 0, 0)]  # Green, Orange, Red
        
        # Initialize device
        self.device = select_device(device)
        
        # Load model
        self.model = self._load_model()
        
        # Create output directory
        self.output_dir = Path('runs/detect')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_model(self):
        """Load YOLOv7 model with trained weights"""
        try:
            # Handle PyTorch 2.6+ weights_only security change
            import torch.serialization
            torch.serialization.add_safe_globals(['numpy._core.multiarray._reconstruct'])
            
            model = attempt_load(self.weights_path, map_location=self.device)
            model.eval()
            
            # Check image size
            self.img_size = check_img_size(self.img_size, s=model.stride.max())
            
            print(f"Model loaded successfully from {self.weights_path}")
            print(f"Device: {self.device}")
            print(f"Image size: {self.img_size}")
            
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
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
    
    def predict_image(self, image_path):
        """
        Run inference on a single image
        
        Args:
            image_path: Path to input image
            
        Returns:
            Tuple of (annotated_image, detection_results)
        """
        if self.model is None:
            return None, "Model not loaded"
        
        try:
            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                return None, "Could not load image"
            
            original_img = img.copy()
            h, w = img.shape[:2]
            
            # Preprocess
            img_tensor, img_resized = self.preprocess_image(img)
            
            # Inference
            with torch.no_grad():
                pred = self.model(img_tensor, augment=False)[0]
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
            
            detection_results = []
            
            # Process detections
            for i, det in enumerate(pred):
                if len(det):
                    # Rescale boxes to original image size
                    det[:, :4] = scale_coords(img_resized.shape, det[:, :4], (h, w)).round()
                    
                    # Draw boxes and labels
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        color = self.colors[int(cls)]
                        
                        # Draw bounding box
                        plot_one_box(xyxy, original_img, label=label, color=color, line_thickness=2)
                        
                        # Store detection info
                        x1, y1, x2, y2 = [int(x) for x in xyxy]
                        detection_results.append({
                            'class': self.names[int(cls)],
                            'confidence': float(conf),
                            'bbox': [x1, y1, x2, y2]
                        })
            
            # Save result
            timestamp = int(time.time())
            output_path = self.output_dir / f'result_{timestamp}.jpg'
            cv2.imwrite(str(output_path), original_img)
            
            return original_img, detection_results
            
        except Exception as e:
            return None, f"Error during inference: {e}"
    
    def predict_video(self, video_path, output_path=None):
        """
        Run inference on video file
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
            
        Returns:
            Path to output video
        """
        if self.model is None:
            return None
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Setup video writer
            if output_path is None:
                timestamp = int(time.time())
                output_path = self.output_dir / f'video_result_{timestamp}.mp4'
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                original_frame = frame.copy()
                img_tensor, img_resized = self.preprocess_image(frame)
                
                with torch.no_grad():
                    pred = self.model(img_tensor, augment=False)[0]
                    pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
                
                # Draw detections
                for i, det in enumerate(pred):
                    if len(det):
                        det[:, :4] = scale_coords(img_resized.shape, det[:, :4], frame.shape).round()
                        
                        for *xyxy, conf, cls in reversed(det):
                            label = f'{self.names[int(cls)]} {conf:.2f}'
                            color = self.colors[int(cls)]
                            plot_one_box(xyxy, frame, label=label, color=color, line_thickness=2)
                
                out.write(frame)
                frame_count += 1
                
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count} frames")
            
            cap.release()
            out.release()
            
            return str(output_path)
            
        except Exception as e:
            print(f"Error processing video: {e}")
            return None

# Global classifier instance
classifier = None

def initialize_classifier():
    """Initialize the classifier with model weights"""
    global classifier
    
    weights_path = 'models/best.pt'
    if not os.path.exists(weights_path):
        return f"Model file not found at {weights_path}. Please place your best.pt file in the models/ directory."
    
    try:
        classifier = RiceClassifier(weights_path=weights_path)
        if classifier.model is None:
            return "Failed to load model. Check if best.pt is valid."
        return "Model loaded successfully!"
    except Exception as e:
        return f"Error initializing classifier: {e}"

def predict_image_interface(image):
    """Gradio interface for image prediction"""
    if classifier is None:
        return None, "Classifier not initialized. Please check model file."
    
    if image is None:
        return None, "Please upload an image."
    
    try:
        # Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            image_pil = image if hasattr(image, 'save') else image
            if hasattr(image_pil, 'save'):
                image_pil.save(tmp_file.name)
            else:
                cv2.imwrite(tmp_file.name, image)
            
            # Run prediction
            result_img, detections = classifier.predict_image(tmp_file.name)
            
            # Clean up temp file
            os.unlink(tmp_file.name)
            
            if result_img is None:
                return None, f"Prediction failed: {detections}"
            
            # Format results
            result_text = f"Detected {len(detections)} objects:\n"
            for i, det in enumerate(detections):
                result_text += f"{i+1}. {det['class']} (confidence: {det['confidence']:.3f})\n"
            
            # Convert BGR to RGB for Gradio display
            result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            
            return result_img_rgb, result_text
            
    except Exception as e:
        return None, f"Error: {e}"

def predict_video_interface(video_file):
    """Gradio interface for video prediction"""
    if classifier is None:
        return None, "Classifier not initialized. Please check model file."
    
    if video_file is None:
        return None, "Please upload a video file."
    
    try:
        # Process video
        output_path = classifier.predict_video(video_file.name)
        
        if output_path is None:
            return None, "Video processing failed."
        
        return output_path, f"Video processed successfully. Output saved to: {output_path}"
        
    except Exception as e:
        return None, f"Error processing video: {e}"

def create_gradio_interface():
    """Create Gradio web interface"""
    
    # Initialize model
    init_status = initialize_classifier()
    
    with gr.Blocks(title="YOLOv7 Rice Quality Classifier") as demo:
        gr.Markdown("# 🌾 YOLOv7 Rice Quality Classifier")
        gr.Markdown("Classify rice grains as **normal**, **broken**, or **crack**")
        
        # Model status
        gr.Markdown(f"**Model Status:** {init_status}")
        
        with gr.Tabs():
            # Image Tab
            with gr.TabItem("Image Classification"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(type="pil", label="Upload Image")
                        image_button = gr.Button("Classify Image", variant="primary")
                    
                    with gr.Column():
                        image_output = gr.Image(label="Detection Result")
                        image_text = gr.Textbox(label="Detection Details", lines=10)
                
                image_button.click(
                    fn=predict_image_interface,
                    inputs=image_input,
                    outputs=[image_output, image_text]
                )
            
            # Video Tab
            with gr.TabItem("Video Classification"):
                with gr.Row():
                    with gr.Column():
                        video_input = gr.File(
                            label="Upload Video (MP4)",
                            file_types=[".mp4", ".avi", ".mov"]
                        )
                        video_button = gr.Button("Process Video", variant="primary")
                    
                    with gr.Column():
                        video_output = gr.File(label="Processed Video")
                        video_text = gr.Textbox(label="Processing Status", lines=5)
                
                video_button.click(
                    fn=predict_video_interface,
                    inputs=video_input,
                    outputs=[video_output, video_text]
                )
            
            # Webcam Tab (Note: Webcam requires special setup)
            with gr.TabItem("Webcam (Instructions)"):
                gr.Markdown("""
                ### Webcam Setup Instructions
                
                For webcam functionality, you need to run YOLOv7 directly:
                
                ```bash
                cd yolov7
                python detect.py --weights ../models/best.pt --source 0 --conf 0.25
                ```
                
                This will open a window showing live webcam feed with detections.
                """)
        
        # Examples
        gr.Markdown("### Example Images")
        gr.Markdown("Upload images of rice grains to test the classifier.")
    
    return demo

def main():
    """Main function to run the application"""
    parser = argparse.ArgumentParser(description='YOLOv7 Rice Classifier Web Demo')
    parser.add_argument('--share', action='store_true', help='Share Gradio interface publicly')
    parser.add_argument('--port', type=int, default=7860, help='Port to run the interface')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the interface')
    
    args = parser.parse_args()
    
    # Check if YOLOv7 is available
    if not Path('./yolov7').exists():
        print("YOLOv7 not found. Please run setup.py first.")
        return
    
    # Create and launch interface
    demo = create_gradio_interface()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True
    )

if __name__ == "__main__":
    main()