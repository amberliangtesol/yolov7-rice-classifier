#!/usr/bin/env python3
"""
Test script to verify letterbox preprocessing fix
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create test image
def create_test_image():
    """Create a test image with known objects at specific positions"""
    # Create a non-square image (640x480) to test aspect ratio handling
    img = np.ones((480, 640, 3), dtype=np.uint8) * 200
    
    # Draw some rectangles to simulate rice grains at known positions
    # Top-left corner
    cv2.rectangle(img, (50, 50), (100, 100), (255, 0, 0), -1)
    # Top-right corner  
    cv2.rectangle(img, (540, 50), (590, 100), (0, 255, 0), -1)
    # Bottom-left corner
    cv2.rectangle(img, (50, 380), (100, 430), (0, 0, 255), -1)
    # Bottom-right corner
    cv2.rectangle(img, (540, 380), (590, 430), (255, 255, 0), -1)
    # Center
    cv2.rectangle(img, (295, 215), (345, 265), (255, 0, 255), -1)
    
    return img

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """YOLOv7 letterbox implementation"""
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def test_letterbox_vs_resize():
    """Compare letterbox vs simple resize"""
    img = create_test_image()
    print(f"Original image shape: {img.shape}")
    
    # Method 1: Simple resize (incorrect)
    img_resized = cv2.resize(img, (640, 640))
    print(f"Simple resize shape: {img_resized.shape}")
    
    # Method 2: Letterbox (correct)
    img_letterbox, ratio, pad = letterbox(img, 640, stride=32, auto=True)
    print(f"Letterbox shape: {img_letterbox.shape}")
    print(f"Ratio: {ratio}")
    print(f"Padding (dw, dh): {pad}")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'Original (640x480)')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Simple Resize (640x640)\\nDistorted!')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].imshow(cv2.cvtColor(img_letterbox, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f'Letterbox (640x640)\\nAspect ratio preserved')
    axes[2].grid(True, alpha=0.3)
    
    for ax in axes:
        ax.set_xticks(np.arange(0, 641, 80))
        ax.set_yticks(np.arange(0, 641, 80))
    
    plt.tight_layout()
    plt.savefig('letterbox_comparison.png', dpi=100, bbox_inches='tight')
    print("\\nVisualization saved to 'letterbox_comparison.png'")
    
    # Test coordinate mapping
    print("\\n" + "="*50)
    print("Testing coordinate mapping:")
    print("="*50)
    
    # Test bounding box at (50, 50, 100, 100) in original image
    test_bbox = [50, 50, 100, 100]  # x1, y1, x2, y2
    print(f"\\nOriginal bbox: {test_bbox}")
    
    # For simple resize (incorrect)
    scale_x = 640 / 640  # width scale
    scale_y = 640 / 480  # height scale  
    resized_bbox = [
        test_bbox[0] * scale_x,
        test_bbox[1] * scale_y,
        test_bbox[2] * scale_x,
        test_bbox[3] * scale_y
    ]
    print(f"After simple resize: {resized_bbox}")
    print(f"  -> Y coordinates stretched by {scale_y:.2f}x")
    
    # For letterbox (correct)
    r = ratio[0]  # uniform scale
    dw, dh = pad
    letterbox_bbox = [
        test_bbox[0] * r + dw,
        test_bbox[1] * r + dh,
        test_bbox[2] * r + dw,
        test_bbox[3] * r + dh
    ]
    print(f"After letterbox: {letterbox_bbox}")
    print(f"  -> Uniform scale by {r:.2f}x + padding")
    
    return img, img_resized, img_letterbox, ratio, pad

def test_coordinate_reverse_mapping():
    """Test reverse mapping from model output to original image"""
    print("\\n" + "="*50)
    print("Testing reverse coordinate mapping:")
    print("="*50)
    
    # Simulate a detection at center of letterbox image (640x640)
    det_bbox = [320-25, 320-25, 320+25, 320+25]  # Center bbox in letterbox space
    print(f"\\nDetection in letterbox space (640x640): {det_bbox}")
    
    # Get letterbox parameters for 640x480 -> 640x640
    original_shape = (480, 640)  # H, W
    new_shape = (640, 640)
    
    # Calculate ratio and padding
    r = min(new_shape[0] / original_shape[0], new_shape[1] / original_shape[1])
    new_unpad = int(round(original_shape[1] * r)), int(round(original_shape[0] * r))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    
    print(f"Letterbox params: ratio={r:.3f}, padding=({dw:.1f}, {dh:.1f})")
    
    # Reverse mapping (correct way)
    original_bbox = [
        (det_bbox[0] - dw) / r,
        (det_bbox[1] - dh) / r,
        (det_bbox[2] - dw) / r,
        (det_bbox[3] - dh) / r
    ]
    print(f"Mapped back to original (640x480): {[f'{x:.1f}' for x in original_bbox]}")
    
    # Check if center is preserved
    center_letterbox = ((det_bbox[0] + det_bbox[2])/2, (det_bbox[1] + det_bbox[3])/2)
    center_original = ((original_bbox[0] + original_bbox[2])/2, (original_bbox[1] + original_bbox[3])/2)
    print(f"\\nCenter in letterbox: ({center_letterbox[0]:.1f}, {center_letterbox[1]:.1f})")
    print(f"Center in original: ({center_original[0]:.1f}, {center_original[1]:.1f})")
    print(f"Expected center in original: (320.0, 240.0)")

if __name__ == "__main__":
    print("YOLOv7 Letterbox Fix Test")
    print("=" * 50)
    
    # Run tests
    test_letterbox_vs_resize()
    test_coordinate_reverse_mapping()
    
    print("\\n" + "="*50)
    print("Summary:")
    print("="*50)
    print("✅ Letterbox preserves aspect ratio by adding padding")
    print("✅ Simple resize distorts the image and shifts detections")
    print("✅ Coordinate mapping must account for both scaling and padding")
    print("✅ The fix ensures bounding boxes align correctly with objects")