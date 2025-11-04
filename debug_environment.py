#!/usr/bin/env python3
"""
Debug script to check environment differences between local and cloud
Add this to your Streamlit app temporarily
"""

import streamlit as st
import cv2
import numpy as np
import torch
import sys
import os
from PIL import Image

def show_environment_info():
    """Display environment information for debugging"""
    
    st.sidebar.header("🔧 Debug Info")
    
    # Python version
    st.sidebar.text(f"Python: {sys.version.split()[0]}")
    
    # Package versions
    st.sidebar.text(f"OpenCV: {cv2.__version__}")
    st.sidebar.text(f"NumPy: {np.__version__}")
    st.sidebar.text(f"Torch: {torch.__version__}")
    st.sidebar.text(f"PIL: {Image.__version__}")
    st.sidebar.text(f"Streamlit: {st.__version__}")
    
    # OpenCV build info
    build_info = cv2.getBuildInformation()
    if "OpenCV modules" in build_info:
        st.sidebar.text("OpenCV build: OK")
    
    # Check if running on cloud
    is_cloud = os.environ.get('STREAMLIT_SHARING_MODE', None) is not None
    st.sidebar.text(f"Cloud deployment: {is_cloud}")
    
    return is_cloud

def test_letterbox_consistency():
    """Test if letterbox behaves the same"""
    test_img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Import letterbox
    sys.path.insert(0, './yolov7')
    from utils.datasets import letterbox
    
    # Test letterbox
    img_out, ratio, pad = letterbox(test_img, 640, stride=32, auto=True)
    
    st.sidebar.text(f"Letterbox test:")
    st.sidebar.text(f"  In: {test_img.shape}")
    st.sidebar.text(f"  Out: {img_out.shape}")
    st.sidebar.text(f"  Ratio: {ratio}")
    st.sidebar.text(f"  Pad: {pad}")
    
    return img_out, ratio, pad

if __name__ == "__main__":
    st.set_page_config(page_title="Environment Debug", layout="wide")
    
    st.title("Environment Debug Info")
    
    is_cloud = show_environment_info()
    
    if st.sidebar.button("Test Letterbox"):
        img_out, ratio, pad = test_letterbox_consistency()
        st.sidebar.success("Letterbox test complete")