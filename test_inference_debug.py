#!/usr/bin/env python3
"""
Debug script to test inference and check coordinate mapping
"""

import cv2
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Import the classifier
from streamlit_app import RiceClassifierStreamlit

def test_inference_with_debug():
    """Test inference with detailed debug output"""
    
    print("="*80)
    print("TESTING YOLOV7 INFERENCE WITH DEBUG OUTPUT")
    print("="*80)
    
    # Initialize classifier
    print("\n[INIT] Loading classifier...")
    classifier = RiceClassifierStreamlit(weights_path='models/best.pt')
    
    # Create a test image with known dimensions
    # Using a non-square image to test letterbox padding
    test_img = np.ones((480, 640, 3), dtype=np.uint8) * 255  # White background
    
    # Add some colored rectangles to simulate rice grains
    # These will help us see if detection boxes align
    cv2.rectangle(test_img, (50, 50), (150, 100), (0, 0, 255), -1)  # Red
    cv2.rectangle(test_img, (200, 200), (300, 250), (0, 255, 0), -1)  # Green  
    cv2.rectangle(test_img, (400, 350), (500, 400), (255, 0, 0), -1)  # Blue
    
    print(f"\n[TEST] Created test image with shape: {test_img.shape}")
    
    # Save test image for reference
    cv2.imwrite('test_input.png', test_img)
    print("[TEST] Saved test image as 'test_input.png'")
    
    # Run inference
    print("\n[TEST] Running inference...")
    result_img, detections = classifier.predict_image(test_img)
    
    print(f"\n[RESULT] Got {len(detections) if detections else 0} detections")
    
    if detections:
        print("\n[DETECTIONS] Details:")
        for i, det in enumerate(detections):
            print(f"  Detection {i+1}:")
            print(f"    Class: {det['class']}")
            print(f"    Confidence: {det['confidence']:.3f}")
            print(f"    BBox: {det['bbox']}")
    
    # Save result if available
    if result_img is not None:
        # Convert RGB to BGR for OpenCV
        result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('test_output.png', result_bgr)
        print("\n[TEST] Saved result image as 'test_output.png'")
    
    print("\n" + "="*80)
    print("TEST COMPLETE - Check console output above for debug info")
    print("="*80)

def test_with_real_image():
    """Test with a real image if available"""
    
    print("\n" + "="*80)
    print("TESTING WITH REAL IMAGE")
    print("="*80)
    
    # Look for sample images
    sample_images = list(Path('.').glob('*.jpg')) + list(Path('.').glob('*.png'))
    
    if not sample_images:
        print("[TEST] No sample images found in current directory")
        return
    
    # Use first found image
    img_path = str(sample_images[0])
    print(f"\n[TEST] Using image: {img_path}")
    
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Failed to load image: {img_path}")
        return
    
    print(f"[TEST] Image shape: {img.shape}")
    
    # Initialize classifier
    classifier = RiceClassifierStreamlit(weights_path='models/best.pt')
    
    # Run inference
    print("\n[TEST] Running inference on real image...")
    result_img, detections = classifier.predict_image(img)
    
    print(f"\n[RESULT] Got {len(detections) if detections else 0} detections")
    
    # Save result
    if result_img is not None:
        output_path = 'real_image_output.png'
        result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, result_bgr)
        print(f"[TEST] Saved result as '{output_path}'")

if __name__ == "__main__":
    # Run synthetic test
    test_inference_with_debug()
    
    # Try real image test
    test_with_real_image()