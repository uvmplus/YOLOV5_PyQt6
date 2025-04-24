# YOLO Detector

A modular GUI application for object detection using YOLOv5.

## Features

- Image detection
- Video detection
- Camera/webcam detection
- User-friendly graphical interface
- Flexible model weights selection

## Project Structure

The project follows a modular architecture for better maintainability and extensibility:

```
yolov5_detector/
├── __init__.py                 # Package initialization
├── main.py                     # Entry point for the application
├── config/                     # Configuration module
│   ├── __init__.py
│   └── settings.py             # Configuration settings and arguments
├── core/                       # Core detection functionality
│   ├── __init__.py
│   ├── detector.py             # YOLOv5 detector implementation
│   └── utils.py                # Utility functions
├── gui/                        # User interface module
│   ├── __init__.py
│   ├── main_window.py          # Main UI window
│   ├── ui_components.py        # Reusable UI components
│   └── styles.py               # UI styling
├── resources/                  # Application resources
└── weights/                    # Model weights storage
```

## Key Design Principles

This refactored version follows several important design principles:

1. **Separation of Concerns**: Each module has a specific responsibility
   - `config`: Configuration management
   - `core`: Detection functionality
   - `gui`: User interface

2. **Modularity**: Components are designed to be independent and reusable
   - `detector.py`: Core detection logic separated from UI
   - `ui_components.py`: Reusable UI elements

3. **Extensibility**: Easy to add new features without modifying existing code
   - New detection methods can be added to the `core` module
   - New UI components can be added to the `gui` module

4. **Maintainability**: Clear organization makes code easier to understand and maintain
   - Well-defined interfaces between modules
   - Consistent coding style and documentation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/yolov5-detector.git
cd yolov5-detector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download YOLOv5 weights (if not already included):
```bash
mkdir -p weights
wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt -O weights/yolov5s.pt
```

## Usage

1. Run the application:
```bash
python main.py
```

2. Using the application:
   - Click "Select Weights" to choose a model weights file
   - Click "Initialize Model" to load the model
   - Use "Image Detection", "Video Detection", or "Camera Detection" accordingly
   - For videos, use the "Pause" and "Stop" buttons to control playback

## Dependencies

- Python 3.6+
- PyQt6
- OpenCV
- PyTorch
- YOLOv5 (included as part of the project)

## How to Extend

### Add a New Detection Method

1. Create a new method in `core/detector.py`
2. Add a UI element in `gui/ui_components.py` 
3. Connect it in `gui/main_window.py`

### Add Support for a New Model

1. Update `core/detector.py` to handle the new model
2. Update configuration in `config/settings.py`

## License

[MIT License](LICENSE)
