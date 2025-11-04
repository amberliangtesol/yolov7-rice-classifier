#!/usr/bin/env python3
"""
Test script to understand scale_coords behavior
"""

import numpy as np
import torch
import sys
from pathlib import Path

# Add YOLOv7 to path
yolo_path = Path('./yolov7')
if yolo_path.exists():
    sys.path.insert(0, str(yolo_path))

from utils.general import scale_coords
from utils.datasets import letterbox
import cv2

def test_scale_coords():
    """Test scale_coords with different scenarios"""
    
    print("="*80)
    print("TESTING SCALE_COORDS BEHAVIOR")
    print("="*80)
    
    # Scenario 1: Image 640x480 -> 640x640 (with padding)
    print("\n[SCENARIO 1] 640x480 -> 640x640 (letterbox)")
    
    # Original image
    original_shape = (480, 640)  # H, W
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Apply letterbox
    img_letterbox, ratio, pad = letterbox(img, 640, stride=32, auto=True)
    print(f"Letterbox output shape: {img_letterbox.shape}")
    print(f"Ratio (from letterbox): {ratio}")  # This is (r, r)
    print(f"Padding (from letterbox): {pad}")  # This is (dw, dh)
    
    # Simulate a detection in letterbox space
    # Center of 640x640 image
    coords = torch.tensor([[320-50, 320-50, 320+50, 320+50]], dtype=torch.float32)  # x1, y1, x2, y2
    print(f"\nDetection in letterbox space: {coords[0].tolist()}")
    
    # Test 1: Let scale_coords calculate itself (ratio_pad=None)
    coords1 = coords.clone()
    img1_shape = (640, 640)  # letterbox shape
    scaled1 = scale_coords(img1_shape, coords1, original_shape, ratio_pad=None)
    print(f"\n[Method 1] ratio_pad=None:")
    print(f"  Result: {scaled1[0].tolist()}")
    
    # Test 2: Pass ratio_pad as documented
    coords2 = coords.clone()
    scaled2 = scale_coords(img1_shape, coords2, original_shape, ratio_pad=(ratio, pad))
    print(f"\n[Method 2] ratio_pad=(ratio, pad) where ratio={ratio}, pad={pad}:")
    print(f"  Result: {scaled2[0].tolist()}")
    
    # Calculate expected result manually
    # For 640x480 -> 640x640:
    # Scale: min(640/480, 640/640) = min(1.333, 1.0) = 1.0
    # New size after scale: 640x480
    # Padding: dw=(640-640)/2=0, dh=(640-480)/2=80
    print(f"\n[EXPECTED] Manual calculation:")
    print(f"  Scale factor: 1.0")
    print(f"  Padding: (0, 80) pixels on top/bottom")
    print(f"  Box (320±50, 320±50) -> (320±50, 240±50)")
    print(f"  Expected: [270, 190, 370, 290]")
    
    # Scenario 2: Image 800x600 -> 640x640
    print("\n" + "="*80)
    print("[SCENARIO 2] 800x600 -> 640x640 (downscale)")
    
    original_shape2 = (600, 800)  # H, W
    img2 = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Apply letterbox
    img_letterbox2, ratio2, pad2 = letterbox(img2, 640, stride=32, auto=True)
    print(f"Letterbox output shape: {img_letterbox2.shape}")
    print(f"Ratio: {ratio2}")
    print(f"Padding: {pad2}")
    
    # Detection at center
    coords3 = torch.tensor([[320-50, 320-50, 320+50, 320+50]], dtype=torch.float32)
    
    # Method 1: ratio_pad=None
    coords3a = coords3.clone()
    scaled3a = scale_coords((640, 640), coords3a, original_shape2, ratio_pad=None)
    print(f"\n[Method 1] ratio_pad=None: {scaled3a[0].tolist()}")
    
    # Method 2: with ratio_pad
    coords3b = coords3.clone()
    scaled3b = scale_coords((640, 640), coords3b, original_shape2, ratio_pad=(ratio2, pad2))
    print(f"[Method 2] ratio_pad=(ratio, pad): {scaled3b[0].tolist()}")

if __name__ == "__main__":
    test_scale_coords()