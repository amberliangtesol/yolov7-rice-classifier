#!/usr/bin/env python3
"""
Verify the letterbox fix is working correctly
"""

import cv2
import numpy as np
from streamlit_app import RiceClassifierStreamlit
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def test_with_different_aspect_ratios():
    """Test with images of different aspect ratios"""
    
    print("="*80)
    print("TESTING LETTERBOX FIX WITH DIFFERENT ASPECT RATIOS")
    print("="*80)
    
    # Initialize classifier
    classifier = RiceClassifierStreamlit(weights_path='models/best.pt')
    
    # Test cases with different aspect ratios
    test_cases = [
        (640, 480, "4:3 ratio"),  # Standard
        (800, 600, "4:3 ratio scaled"),
        (1920, 1080, "16:9 ratio"),
        (640, 640, "1:1 square"),
        (480, 640, "3:4 portrait"),
    ]
    
    for width, height, description in test_cases:
        print(f"\n[TEST] {description}: {width}x{height}")
        
        # Create test image
        img = np.ones((height, width, 3), dtype=np.uint8) * 200
        
        # Add some test rectangles
        cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), -1)
        cv2.rectangle(img, (width-150, height-150), (width-50, height-50), (0, 255, 0), -1)
        
        # Run inference
        result_img, detections = classifier.predict_image(img)
        
        print(f"  Detections: {len(detections) if detections else 0}")
        
        if detections and len(detections) > 0:
            print("  First detection bbox:", detections[0]['bbox'])
    
    print("\n" + "="*80)
    print("If bounding boxes appear at consistent positions relative to objects,")
    print("the letterbox fix is working correctly.")
    print("="*80)

if __name__ == "__main__":
    test_with_different_aspect_ratios()