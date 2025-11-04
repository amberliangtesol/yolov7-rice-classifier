#!/usr/bin/env python3
"""
Direct test using YOLOv7's detect.py to verify if issue is in our implementation
"""

import subprocess
import os

def test_with_yolov7_detect():
    """Test using YOLOv7's original detect.py"""
    
    print("="*80)
    print("Testing with YOLOv7's original detect.py")
    print("="*80)
    
    # Check if test image exists
    test_images = ['test_rice.jpg', 'test_rice.png', 'sample_rice.jpg']
    test_img = None
    
    for img in test_images:
        if os.path.exists(img):
            test_img = img
            break
    
    if not test_img:
        print("No test image found. Please provide a rice image.")
        return
    
    # Run YOLOv7 detect.py
    cmd = [
        'python', 'yolov7/detect.py',
        '--weights', 'models/best.pt',
        '--source', test_img,
        '--img-size', '640',
        '--conf', '0.25',
        '--device', 'cpu',
        '--save-txt',
        '--save-conf',
        '--name', 'test_detect'
    ]
    
    print(f"\nRunning command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("\nOutput:")
    print(result.stdout)
    
    if result.stderr:
        print("\nErrors:")
        print(result.stderr)

if __name__ == "__main__":
    test_with_yolov7_detect()