#!/usr/bin/env python3
"""
Setup script for YOLOv7 Rice Quality Classifier
This script clones YOLOv7 repository and installs dependencies
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

def check_gpu():
    """Check if GPU is available"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
            return True
        else:
            print("No GPU detected. Will use CPU.")
            return False
    except ImportError:
        print("PyTorch not installed yet.")
        return False

def install_pytorch():
    """Install PyTorch based on system configuration"""
    print("\n=== Installing PyTorch ===")
    
    # Check if CUDA is available
    try:
        # Try to detect CUDA
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True)
        cuda_available = result.returncode == 0
    except:
        cuda_available = False
    
    if cuda_available:
        print("CUDA detected. Installing PyTorch with CUDA support...")
        pytorch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    else:
        print("No CUDA detected. Installing CPU-only PyTorch...")
        pytorch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    
    return run_command(pytorch_cmd)

def clone_yolov7():
    """Clone YOLOv7 repository"""
    print("\n=== Cloning YOLOv7 Repository ===")
    
    yolo_dir = Path("yolov7")
    if yolo_dir.exists():
        print("YOLOv7 directory already exists. Removing...")
        shutil.rmtree(yolo_dir)
    
    # Clone YOLOv7
    clone_cmd = "git clone https://github.com/WongKinYiu/yolov7.git"
    if not run_command(clone_cmd):
        return False
    
    # Install YOLOv7 requirements
    print("Installing YOLOv7 requirements...")
    yolo_requirements = yolo_dir / "requirements.txt"
    if yolo_requirements.exists():
        return run_command(f"pip install -r {yolo_requirements}")
    
    return True

def install_requirements():
    """Install project requirements"""
    print("\n=== Installing Project Requirements ===")
    
    requirements_file = Path("requirements.txt")
    if requirements_file.exists():
        return run_command(f"pip install -r {requirements_file}")
    else:
        print("requirements.txt not found!")
        return False

def setup_directories():
    """Create necessary directories"""
    print("\n=== Setting up Directories ===")
    
    directories = [
        "models",
        "runs/detect",
        "data/images",
        "data/videos"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    return True

def create_colab_notebook():
    """Create Google Colab notebook"""
    print("\n=== Creating Colab Notebook ===")
    
    notebook_content = '''
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# YOLOv7 Rice Quality Classifier\\n",
    "## Setup and Run on Google Colab\\n",
    "\\n",
    "This notebook sets up and runs the YOLOv7 rice quality classifier on Google Colab."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Check GPU availability\\n",
    "!nvidia-smi\\n",
    "\\n",
    "import torch\\n",
    "print(f\\"PyTorch version: {torch.__version__}\\")\\n",
    "print(f\\"CUDA available: {torch.cuda.is_available()}\\")\\n",
    "if torch.cuda.is_available():\\n",
    "    print(f\\"GPU: {torch.cuda.get_device_name(0)}\\")\\n",
    "    print(f\\"CUDA version: {torch.version.cuda}\\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Clone the project repository (replace with your repo URL)\\n",
    "!git clone https://github.com/your-username/yolov7-rice-classifier.git\\n",
    "%cd yolov7-rice-classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Run setup script\\n",
    "!python setup.py"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Upload your best.pt model file\\n",
    "from google.colab import files\\n",
    "\\n",
    "print(\\"Please upload your best.pt model file:\\")\\n",
    "uploaded = files.upload()\\n",
    "\\n",
    "# Move to models directory\\n",
    "import shutil\\n",
    "for filename in uploaded.keys():\\n",
    "    if filename.endswith('.pt'):\\n",
    "        shutil.move(filename, 'models/best.pt')\\n",
    "        print(f\\"Moved {filename} to models/best.pt\\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install additional dependencies for Colab\\n",
    "!pip install gradio"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Launch the Gradio interface\\n",
    "!python rice_classifier_app.py --share"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Alternative: Direct YOLOv7 Detection\\n",
    "\\n",
    "You can also run YOLOv7 directly for batch processing:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Direct inference with YOLOv7\\n",
    "%cd yolov7\\n",
    "\\n",
    "# For image detection\\n",
    "!python detect.py --weights ../models/best.pt --source ../data/images --conf 0.25\\n",
    "\\n",
    "# For webcam (if available)\\n",
    "# !python detect.py --weights ../models/best.pt --source 0 --conf 0.25\\n",
    "\\n",
    "# For video\\n",
    "# !python detect.py --weights ../models/best.pt --source ../data/videos/test_video.mp4 --conf 0.25"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
'''
    
    with open("rice_classifier_colab.ipynb", "w") as f:
        f.write(notebook_content)
    
    print("Created rice_classifier_colab.ipynb")
    return True

def main():
    """Main setup function"""
    print("🌾 YOLOv7 Rice Quality Classifier Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required")
        return False
    
    print(f"Python version: {sys.version}")
    
    # Setup steps
    steps = [
        ("Setting up directories", setup_directories),
        ("Installing PyTorch", install_pytorch),
        ("Cloning YOLOv7", clone_yolov7),
        ("Installing requirements", install_requirements),
        ("Creating Colab notebook", create_colab_notebook)
    ]
    
    for step_name, step_func in steps:
        print(f"\\n{'='*20} {step_name} {'='*20}")
        if not step_func():
            print(f"❌ Failed: {step_name}")
            return False
        print(f"✅ Completed: {step_name}")
    
    # Final GPU check
    print("\\n" + "="*50)
    check_gpu()
    
    print("\\n🎉 Setup completed successfully!")
    print("\\nNext steps:")
    print("1. Place your 'best.pt' model file in the 'models/' directory")
    print("2. Run the application:")
    print("   python rice_classifier_app.py")
    print("\\nFor Google Colab:")
    print("   Upload rice_classifier_colab.ipynb to Colab and follow the instructions")
    
    return True

if __name__ == "__main__":
    main()