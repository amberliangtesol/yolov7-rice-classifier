# YOLOv7 Rice Quality Classifier - Claude Code Session Notes

## Project Overview
YOLOv7-based rice quality classification web application with Streamlit interface. Classifies rice grains into three categories: normal, broken, and crack.

## Key Features
- **Image Classification**: Upload images for rice quality analysis
- **Video Processing**: Process videos with H.264 conversion for browser compatibility
- **Live Camera**: Real-time classification (requires local YOLOv7 setup)
- **Batch Analysis**: Process multiple files and generate statistics

## Recent Development Sessions

### Video Processing Fix (November 2024)
**Problem**: Video preview failing in Streamlit Cloud deployment with "Video source error"

**Root Cause**: Missing ffmpeg/ffprobe in cloud environment
- H.264 conversion failures due to missing ffmpeg
- Video metadata analysis failing
- Browser compatibility issues with unconverted video formats

**Solution Implemented**:
1. **packages.txt Update**: Added `ffmpeg` to packages.txt for automatic installation in Streamlit Cloud
2. **Enhanced Error Handling**: Graceful fallback to download buttons when conversion fails
3. **UI Cleanup**: Removed technical status messages for cleaner user experience

**Files Modified**:
- `packages.txt`: Added ffmpeg package
- `streamlit_app.py`: Enhanced H.264 conversion and error handling

### UI Improvements
- Removed verbose technical messages during video processing
- Cleaner user interface with less visual clutter
- Maintained functionality while improving user experience

## Technical Stack
- **Deep Learning**: YOLOv7 for object detection
- **Web Framework**: Streamlit
- **Video Processing**: OpenCV, ffmpeg
- **Deployment**: Streamlit Cloud
- **Language**: Python

## Project Structure
```
yolov7-rice-classifier/
├── streamlit_app.py          # Main Streamlit application
├── rice_classifier_app.py    # Gradio alternative interface
├── packages.txt              # System packages for deployment
├── requirements.txt          # Python dependencies
├── models/                   # Model weights directory
│   └── best.pt              # Trained YOLOv7 weights
├── yolov7/                  # YOLOv7 source code
└── runs/                    # Output directory for results
```

## Commands Reference

### Local Development
```bash
# Start Streamlit app
streamlit run streamlit_app.py

# Start Gradio app
python rice_classifier_app.py

# Direct YOLOv7 inference
cd yolov7
python detect.py --weights ../models/best.pt --source 0 --conf 0.25
```

### Testing
```bash
# Test video processing (requires ffmpeg)
# Upload video files through web interface

# Test image classification
# Upload images through web interface
```

### Deployment
- **Platform**: Streamlit Cloud
- **Auto-deploy**: Connected to GitHub repository
- **Dependencies**: Automatically installed via packages.txt and requirements.txt

## Known Issues & Solutions

### Issue: ffmpeg Not Available Locally
**Symptoms**: 
- `ffprobe 失敗：[Errno 2] No such file or directory: 'ffprobe'`
- H.264 conversion failures in local testing

**Solution**: 
- Install ffmpeg locally: `brew install ffmpeg` (macOS) or equivalent
- Or use Streamlit Cloud deployment where ffmpeg is auto-installed

### Issue: Video Processing UI Clutter
**Symptoms**: Too many technical status messages
**Solution**: Cleaned up UI by removing verbose technical feedback

## Performance Notes
- Model loading time: ~2-3 seconds on first run
- Image processing: Near real-time
- Video processing: Depends on length and resolution
- H.264 conversion: Adds ~30% processing time but improves browser compatibility

## Future Improvements
- [ ] Add confidence threshold controls
- [ ] Implement batch download for results
- [ ] Add model performance metrics
- [ ] Enhance mobile responsiveness
- [ ] Add export formats (CSV, JSON)

## Links
- **Live App**: [Streamlit Cloud Deployment URL]
- **Repository**: https://github.com/amberliangtesol/yolov7-rice-classifier
- **YOLOv7 Source**: https://github.com/WongKinYiu/yolov7

---
*Last updated: November 2024*
*Generated with Claude Code assistance*