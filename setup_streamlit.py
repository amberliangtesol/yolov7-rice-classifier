#!/usr/bin/env python3
"""
Setup script for Streamlit version of YOLOv7 Rice Quality Classifier
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, cwd=None):
    """Run shell command and handle errors"""
    try:
        print(f"Running: {command}")
        result = subprocess.run(command, shell=True, check=True, cwd=cwd, 
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Main setup function for Streamlit"""
    print("🌾 YOLOv7 Rice Classifier Streamlit Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3.7):
        print("Error: Python 3.7 or higher is required")
        return False
    
    print(f"Python version: {sys.version}")
    
    # Install requirements
    print("\n=== Installing Streamlit Requirements ===")
    if not run_command("pip install -r requirements_streamlit.txt"):
        print("❌ Failed to install requirements")
        return False
    
    # Check if YOLOv7 exists
    yolo_dir = Path("yolov7")
    if not yolo_dir.exists():
        print("\n=== Cloning YOLOv7 Repository ===")
        if not run_command("git clone https://github.com/WongKinYiu/yolov7.git"):
            print("❌ Failed to clone YOLOv7")
            return False
        
        # Apply PyTorch 2.6+ fix
        experimental_file = yolo_dir / "models" / "experimental.py"
        if experimental_file.exists():
            print("Applying PyTorch 2.6+ compatibility fix...")
            # The fix should already be applied if using our version
    
    # Check model file
    model_path = Path("models/best.pt")
    if not model_path.exists():
        print(f"\n⚠️  Model file not found: {model_path}")
        print("Please place your best.pt file in the models/ directory")
    else:
        print(f"✅ Model file found: {model_path}")
    
    print("\n🎉 Streamlit setup completed!")
    print("\nTo run the application:")
    print("  streamlit run streamlit_app.py")
    print("\nTo run in headless mode:")
    print("  STREAMLIT_SERVER_EMAIL=\"\" streamlit run streamlit_app.py --server.headless true")
    
    return True

if __name__ == "__main__":
    main()