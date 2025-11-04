#!/usr/bin/env python3
"""
Streamlit Cloud compatibility fix
Key issue: Image display and coordinate mapping differences between local and cloud
"""

import numpy as np
import cv2
from PIL import Image

def ensure_cloud_compatibility(image, detections):
    """
    Ensure image and detections are compatible with Streamlit Cloud display
    
    Args:
        image: numpy array (BGR or RGB)
        detections: list of detection dictionaries
    
    Returns:
        PIL Image with detections drawn correctly
    """
    
    # Ensure image is RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert BGR to RGB if needed
        if isinstance(image, np.ndarray):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
    else:
        image_rgb = image
    
    # Convert to PIL for consistent display
    pil_image = Image.fromarray(image_rgb.astype('uint8'))
    
    # Draw detections using PIL (more reliable on cloud)
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(pil_image)
    
    # Try to use a font, fallback to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 15)
    except:
        font = ImageFont.load_default()
    
    colors = {
        'white_rice': (255, 255, 255),
        'thi_rice': (255, 215, 0),
        'brown_rice': (139, 69, 19),
        'black_rice': (0, 0, 0)
    }
    
    for det in detections:
        bbox = det['bbox']
        class_name = det['class']
        confidence = det['confidence']
        
        # Ensure coordinates are integers
        x1, y1, x2, y2 = [int(x) for x in bbox]
        
        # Get color
        color = colors.get(class_name, (255, 0, 0))
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        # Draw label
        label = f"{class_name} {confidence:.2f}"
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], 
                      fill=color)
        draw.text((x1, y1), label, fill=(255, 255, 255), font=font)
    
    return pil_image

def fix_streamlit_display(result_img, detection_results):
    """
    Fix for Streamlit Cloud display issues
    """
    # Always return PIL Image for consistency
    if isinstance(result_img, np.ndarray):
        # Ensure RGB
        if len(result_img.shape) == 3:
            if result_img.shape[2] == 3:
                # Assume BGR if numpy array from OpenCV
                result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            else:
                result_img_rgb = result_img
        else:
            result_img_rgb = result_img
        
        # Convert to PIL
        pil_img = Image.fromarray(result_img_rgb.astype('uint8'))
    else:
        pil_img = result_img
    
    return pil_img

# Test coordinate mapping
def test_coordinate_mapping():
    """Test if coordinates map correctly"""
    # Create test image
    img = np.ones((640, 640, 3), dtype=np.uint8) * 255
    
    # Test detection
    test_det = {
        'class': 'white_rice',
        'confidence': 0.95,
        'bbox': [100, 100, 200, 200]
    }
    
    # Apply fix
    result = ensure_cloud_compatibility(img, [test_det])
    
    return result

if __name__ == "__main__":
    print("Streamlit Cloud Fix Module")
    print("This module ensures consistent image display between local and cloud deployments")
    
    # Test
    test_img = test_coordinate_mapping()
    test_img.save("cloud_fix_test.png")
    print("Test image saved as cloud_fix_test.png")