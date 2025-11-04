#!/usr/bin/env python3
"""
Final alignment fix for YOLOv7 Rice Classifier
This script contains the corrected coordinate transformation logic
"""

import numpy as np
import cv2
import torch

def fixed_preprocess_image(img, img_size, letterbox_func, model_stride, device):
    """
    Fixed preprocessing that properly tracks letterbox parameters
    """
    # Apply letterbox and KEEP all return values
    img_letterbox, ratio, pad = letterbox_func(img, img_size, stride=int(model_stride), auto=True)
    
    # Convert BGR to RGB and transpose
    img_rgb = img_letterbox[:, :, ::-1].transpose(2, 0, 1)
    img_rgb = np.ascontiguousarray(img_rgb)
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img_rgb).to(device)
    img_tensor = img_tensor.float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)
    
    # Return all needed values for coordinate transformation
    return img_tensor, img_letterbox, ratio, pad

def fixed_predict_image(image, model, preprocess_func, scale_coords_func, 
                        conf_thres, iou_thres, names, colors, nms_func):
    """
    Fixed prediction with proper coordinate transformation
    """
    # Handle PIL Image input
    if hasattr(image, 'convert'):  # PIL Image
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        img = image
    
    original_img = img.copy()
    original_shape = img.shape  # Keep full shape for scale_coords
    
    # Preprocess WITH letterbox tracking
    img_tensor, img_letterbox, ratio, pad = preprocess_func(img)
    
    # Inference
    with torch.no_grad():
        pred = model(img_tensor, augment=False)[0]
        pred = nms_func(pred, conf_thres, iou_thres)
    
    detection_results = []
    
    # Process detections
    for i, det in enumerate(pred):
        if len(det):
            # Critical: Use correct shapes for scale_coords
            # img_tensor.shape[2:] = letterbox size (e.g., 640x640)
            # original_shape = original image shape WITH channels
            det[:, :4] = scale_coords_func(
                img_tensor.shape[2:],  # letterbox shape
                det[:, :4],            # coordinates
                original_shape         # original shape (H, W, C)
            ).round()
            
            # Draw on original image
            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = [int(x) for x in xyxy]
                
                # Draw box
                color = colors[int(cls)]
                cv2.rectangle(original_img, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f'{names[int(cls)]} {conf:.2f}'
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
                
                cv2.rectangle(original_img,
                            (x1, label_y - label_size[1] - 3),
                            (x1 + label_size[0], label_y + 3),
                            color, -1)
                cv2.putText(original_img, label, (x1, label_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                detection_results.append({
                    'class': names[int(cls)],
                    'confidence': float(conf),
                    'bbox': [x1, y1, x2, y2]
                })
    
    # Convert to RGB for display
    result_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    return result_img_rgb, detection_results

if __name__ == "__main__":
    print("Final Alignment Fix Module")
    print("="*50)
    print("Key fixes:")
    print("1. Properly track letterbox ratio and padding")
    print("2. Pass full original shape to scale_coords")
    print("3. Use consistent coordinate transformation")
    print("4. Draw on original image, not letterbox")
    print("="*50)