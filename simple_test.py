#!/usr/bin/env python3
"""
Simple test script to check if YOLOv7 can be imported and loaded
"""

import os
import sys
import torch
from pathlib import Path

print("🌾 Testing YOLOv7 Rice Classifier Setup")
print("=" * 50)

# Add YOLOv7 to path
yolo_path = Path('./yolov7')
if yolo_path.exists():
    sys.path.append(str(yolo_path))
    print(f"✅ YOLOv7 path added: {yolo_path.absolute()}")
else:
    print("❌ YOLOv7 directory not found")
    sys.exit(1)

# Test PyTorch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Test YOLOv7 imports
try:
    from models.experimental import attempt_load
    print("✅ models.experimental imported successfully")
    
    from utils.general import check_img_size, non_max_suppression, scale_coords
    print("✅ utils.general imported successfully")
    
    from utils.plots import plot_one_box
    print("✅ utils.plots imported successfully")
    
    from utils.torch_utils import select_device
    print("✅ utils.torch_utils imported successfully")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test model loading
try:
    device = select_device('')
    print(f"✅ Device selected: {device}")
    
    model_path = 'models/best.pt'
    if os.path.exists(model_path):
        print(f"✅ Model file found: {model_path}")
        
        # Try to load the model
        model = attempt_load(model_path, map_location=device)
        print(f"✅ Model loaded successfully!")
        print(f"Model type: {type(model)}")
        
        # Test image size
        img_size = check_img_size(640, s=model.stride.max())
        print(f"✅ Image size checked: {img_size}")
        
    else:
        print(f"❌ Model file not found: {model_path}")
        
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

print("\n🎉 All tests passed! YOLOv7 setup is working correctly.")
print("You can now run the full application with: python rice_classifier_app.py")