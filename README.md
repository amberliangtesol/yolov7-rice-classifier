# 🌾 YOLOv7 Rice Quality Classifier

A web-based application for classifying rice grain quality using YOLOv7 deep learning model. The system can detect and classify rice grains into three categories: **normal**, **broken**, and **crack**.

## 📁 Project Structure

```
yolov7-rice-classifier/
├── models/                    # Place your best.pt model here
│   └── best.pt               # Your trained YOLOv7 model (you need to add this)
├── runs/detect/              # Output detection results
├── data/                     # Sample data (optional)
│   ├── images/              # Test images
│   └── videos/              # Test videos
├── yolov7/                   # YOLOv7 repository (auto-downloaded)
├── rice_classifier_app.py    # Main Gradio web application
├── setup.py                  # Setup script
├── requirements.txt          # Python dependencies
├── rice_classifier_colab.ipynb  # Google Colab notebook
└── README.md                 # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone this repository or download the files
cd yolov7-rice-classifier

# Run setup script (installs dependencies and clones YOLOv7)
python setup.py
```

### 2. Add Your Model

**⚠️ IMPORTANT: Place your trained `best.pt` model file in the `models/` directory**

```bash
# Copy your model file
cp /path/to/your/best.pt models/best.pt
```

### 3. Run the Application

```bash
# Launch Gradio web interface
python rice_classifier_app.py

# Optional arguments:
python rice_classifier_app.py --share  # Create public link
python rice_classifier_app.py --port 8080  # Custom port
```

The web interface will open at `http://localhost:7860`

## 🌐 Features

### Web Interface (Gradio)
- **Image Classification**: Upload single images for detection
- **Video Processing**: Process MP4/AVI video files
- **Real-time Results**: View detection boxes and confidence scores
- **Export Results**: Automatically save results to `runs/detect/`

### Supported Inputs
- **Images**: JPG, PNG, JPEG
- **Videos**: MP4, AVI, MOV
- **Webcam**: Live camera feed (command line only)

### Detection Classes
1. **Normal**: Healthy rice grains
2. **Broken**: Damaged/broken grains  
3. **Crack**: Grains with cracks

## 💻 Google Colab Usage

1. Upload `rice_classifier_colab.ipynb` to Google Colab
2. Follow the notebook instructions
3. Upload your `best.pt` model when prompted
4. Run all cells to launch the interface

## 🖥️ Command Line Usage

For direct YOLOv7 detection without web interface:

```bash
cd yolov7

# Detect on single image
python detect.py --weights ../models/best.pt --source image.jpg --conf 0.25

# Detect on folder of images
python detect.py --weights ../models/best.pt --source ../data/images/ --conf 0.25

# Webcam detection
python detect.py --weights ../models/best.pt --source 0 --conf 0.25

# Video detection
python detect.py --weights ../models/best.pt --source video.mp4 --conf 0.25

# Batch processing
python detect.py --weights ../models/best.pt --source ../data/images/ --save-txt --save-conf
```

## ⚙️ Configuration

### Model Parameters
- **Confidence Threshold**: 0.25 (adjustable in code)
- **IoU Threshold**: 0.45 (adjustable in code)
- **Image Size**: 640x640 (adjustable in code)

### GPU Support
- Automatically detects and uses GPU if available
- Falls back to CPU if no GPU detected
- CUDA 11.8+ recommended for GPU support

## 📊 Model Information

- **Architecture**: YOLOv7
- **Classes**: 3 (normal, broken, crack)
- **Input Size**: 640x640
- **Format**: PyTorch (.pt)

## 🛠️ Development

### File Descriptions

- `rice_classifier_app.py`: Main application with Gradio interface
- `setup.py`: Automated setup script
- `requirements.txt`: Python package dependencies
- `rice_classifier_colab.ipynb`: Google Colab notebook template

### Dependencies

- PyTorch >= 1.7.0
- YOLOv7 (auto-downloaded)
- Gradio >= 3.50.0
- OpenCV >= 4.1.1
- NumPy, Matplotlib, PIL

## 🚨 Troubleshooting

### Common Issues

1. **Model not found error**
   - Ensure `best.pt` is in `models/` directory
   - Check file permissions

2. **CUDA out of memory**
   - Reduce image size in code
   - Use CPU mode: set `device='cpu'`

3. **Import errors**
   - Run `python setup.py` again
   - Check Python version (3.7+ required)

4. **Webcam not working**
   - Try different camera indices (0, 1, 2)
   - Check camera permissions

### Performance Tips

- Use GPU for faster inference
- Reduce confidence threshold for more detections
- Increase confidence threshold for fewer false positives
- Batch process multiple images for efficiency

## 📝 Usage Examples

### Python API

```python
from rice_classifier_app import RiceClassifier

# Initialize classifier
classifier = RiceClassifier(weights_path='models/best.pt')

# Predict single image
result_img, detections = classifier.predict_image('test_image.jpg')

# Process video
output_path = classifier.predict_video('test_video.mp4')
```

### Web Interface

1. Open `http://localhost:7860`
2. Navigate to "Image Classification" tab
3. Upload an image
4. Click "Classify Image"
5. View results with bounding boxes and confidence scores

## 📈 Model Training

This classifier expects a YOLOv7 model trained on rice grain dataset with:
- 3 classes: normal, broken, crack
- COCO format annotations
- Recommended training epochs: 100+
- Image size: 640x640

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

## 📄 License

This project is for educational and research purposes. Please respect YOLOv7 license terms.

## 🙏 Acknowledgments

- [YOLOv7](https://github.com/WongKinYiu/yolov7) by WongKinYiu
- [Gradio](https://gradio.app/) for web interface
- Rice quality dataset contributors

---

**Need help?** Check the troubleshooting section or open an issue with your specific problem.