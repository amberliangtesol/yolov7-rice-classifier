# 🌾 YOLOv7 Rice Quality Classifier - Streamlit Version

A Streamlit web application for classifying rice grain quality using YOLOv7 deep learning model. The system can detect and classify rice grains into three categories: **normal**, **broken**, and **crack**.

## 🚀 Quick Start (Streamlit)

### 1. Setup Environment

```bash
# Install Streamlit requirements
pip install -r requirements_streamlit.txt

# Or run the setup script
python setup_streamlit.py
```

### 2. Add Your Model

Place your trained `best.pt` model file in the `models/` directory:

```bash
cp /path/to/your/best.pt models/best.pt
```

### 3. Run Streamlit Application

```bash
# Basic launch
streamlit run streamlit_app.py

# Headless mode (no browser popup)
STREAMLIT_SERVER_EMAIL="" streamlit run streamlit_app.py --server.headless true

# Custom port
streamlit run streamlit_app.py --server.port 8080
```

The web interface will open at `http://localhost:8502`

## 🌟 Streamlit Features

### 📱 User Interface
- **Clean Design**: Modern, responsive interface with custom styling
- **Tabbed Navigation**: Separate tabs for different functions
- **Real-time Configuration**: Adjust detection parameters via sidebar
- **Progress Indicators**: Visual feedback during processing

### 🔧 Functionality
- **Image Upload**: Drag-and-drop or browse for images
- **Live Detection**: Real-time rice grain classification
- **Detection Statistics**: Count by class with visual metrics
- **Detailed Results**: Expandable detection information
- **Confidence Controls**: Adjustable thresholds via sliders

### 📊 Display Features
- **Side-by-side Comparison**: Original vs. detected images
- **Color-coded Classes**: 
  - 🟢 Normal (green)
  - 🟠 Broken (orange) 
  - 🔴 Crack (red)
- **Bounding Box Visualization**: Clear detection overlays
- **Export Results**: Automatic saving to runs/detect/

## 📋 Interface Tabs

### 1. 📷 Image Upload
- Upload PNG, JPG, JPEG files
- Real-time inference display
- Detection summary statistics
- Detailed results with confidence scores

### 2. 🎥 Video Upload
- Video file upload interface
- Command-line processing instructions
- Batch processing guidance

### 3. 📹 Webcam
- Live camera feed instructions
- YOLOv7 direct command examples
- Real-time detection setup guide

### 4. ℹ️ Instructions
- Complete usage guide
- Settings explanation
- Command-line options
- System information display

## ⚙️ Configuration

### Streamlit Settings (`.streamlit/config.toml`)
```toml
[server]
headless = true
port = 8502
maxUploadSize = 1028

[theme]
primaryColor = "#2E7D32"
backgroundColor = "#FFFFFF"
```

### Detection Parameters
- **Confidence Threshold**: 0.1 - 1.0 (default: 0.25)
- **IoU Threshold**: 0.1 - 1.0 (default: 0.45)
- **Image Size**: 640x640 (automatic)

## 🔄 Comparison: Streamlit vs Gradio

| Feature | Streamlit | Gradio |
|---------|-----------|---------|
| **Interface** | Modern, customizable | Simple, functional |
| **Performance** | Fast, cached | Good |
| **Customization** | High (CSS, themes) | Medium |
| **Deployment** | Multiple options | Share links |
| **Learning Curve** | Medium | Easy |
| **Mobile Support** | Excellent | Good |

## 🌐 Deployment Options

### 1. Streamlit Cloud (Free)
```bash
# Push to GitHub
git add .
git commit -m "Add Streamlit rice classifier"
git push origin main

# Deploy at streamlit.io
# Connect GitHub repo
# Select streamlit_app.py as main file
```

### 2. Heroku
```bash
# Create Procfile
echo "web: streamlit run streamlit_app.py --server.port \$PORT --server.headless true" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

### 3. Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements_streamlit.txt .
RUN pip install -r requirements_streamlit.txt

COPY . .
EXPOSE 8502

CMD ["streamlit", "run", "streamlit_app.py", "--server.headless", "true"]
```

### 4. Local Network
```bash
# Run on all interfaces
streamlit run streamlit_app.py --server.address 0.0.0.0
```

## 🛠️ Development Features

### Caching
- `@st.cache_resource` for model loading
- `@st.cache_data` for data processing
- Persistent model in memory

### Session State
- Configuration persistence
- Upload history
- User preferences

### Error Handling
- Graceful failure modes
- Informative error messages
- Fallback options

## 📈 Performance Tips

### Optimization
- Model caching reduces load time
- Image preprocessing optimization
- Batch processing for multiple images

### Memory Management
- Automatic cleanup of temporary files
- Efficient tensor operations
- GPU memory optimization

## 🔧 Troubleshooting

### Common Issues

1. **Streamlit not starting**
   ```bash
   # Check if port is available
   lsof -i :8502
   
   # Use different port
   streamlit run streamlit_app.py --server.port 8080
   ```

2. **Model loading errors**
   ```bash
   # Check model file
   ls -la models/best.pt
   
   # Verify PyTorch version
   python -c "import torch; print(torch.__version__)"
   ```

3. **Upload size limits**
   ```bash
   # Increase limit in config.toml
   [server]
   maxUploadSize = 2048
   ```

## 📱 Mobile Responsiveness

The Streamlit interface is fully responsive and works well on:
- 📱 Mobile phones
- 📱 Tablets  
- 💻 Desktop computers
- 🖥️ Large screens

## 🎨 Customization

### Themes
Edit `.streamlit/config.toml` to customize:
- Primary colors
- Background colors
- Font styles
- Layout options

### CSS Styling
Add custom CSS in `streamlit_app.py`:
```python
st.markdown("""
<style>
    .custom-class {
        /* Your styles here */
    }
</style>
""", unsafe_allow_html=True)
```

## 📊 Analytics

Streamlit provides built-in analytics:
- User engagement metrics
- Feature usage statistics
- Performance monitoring
- Error tracking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add Streamlit-specific improvements
4. Test thoroughly
5. Submit pull request

## 📄 License

This project is for educational and research purposes. Please respect YOLOv7 and Streamlit license terms.

---

**Need help?** Check the troubleshooting section or create an issue with your specific problem.