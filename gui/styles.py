"""
UI styling constants and functions for the YOLOv5 Detector application.
"""

# Main application stylesheet
MAIN_STYLE = """
/* Global Styles */
QMainWindow, QWidget {
    background-color: #f8f9fa;
    font-family: 'Segoe UI', Arial, sans-serif;
}

/* Button Styling */
QPushButton {
    background-color: #4361ee;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 12px 20px;
    font-size: 14px;
    font-weight: 500;
    min-height: 40px;
    min-width: 150px;
    margin: 5px 0px;
}
QPushButton:hover {
    background-color: #3a56d4;
}
QPushButton:pressed {
    background-color: #2e46be;
}
QPushButton:disabled {
    background-color: #b3b3b3;
    color: #e0e0e0;
}

/* Specialized Button Styles */
#pushButton_weights, #pushButton_init {
    background-color: #5a189a;
}
#pushButton_weights:hover, #pushButton_init:hover {
    background-color: #4c1184;
}

#pushButton_stop {
    background-color: #f9c74f;
    color: #333333;
}
#pushButton_stop:hover {
    background-color: #f3b846;
}

#pushButton_finish {
    background-color: #e63946;
}
#pushButton_finish:hover {
    background-color: #d62b39;
}

/* Display Area */
QLabel#label {
    background-color: #212529;
    border: 2px solid #343a40;
    border-radius: 8px;
}

QTextBrowser#status_display {
    background-color: #ffffff;
    border: 1px solid #dee2e6;
    border-radius: 6px;
    padding: 8px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 13px;
    color: #495057;
}

/* Groupbox Styling */
QGroupBox {
    border: 1px solid #ced4da;
    border-radius: 8px;
    margin-top: 20px;
    font-weight: bold;
    color: #495057;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}
"""

# Frame styles
DISPLAY_FRAME_STYLE = """
QFrame {
    background-color: #343a40;
    border-radius: 10px;
    padding: 5px;
}
"""

STATUS_FRAME_STYLE = """
QFrame {
    background-color: white;
    border-radius: 10px;
    border: 1px solid #e9ecef;
}
"""

# Label styles
TITLE_LABEL_STYLE = """
font-size: 22px; 
font-weight: bold; 
color: #212529;
margin-bottom: 15px;
padding: 10px;
"""

STATUS_LABEL_STYLE = """
font-weight: bold;
color: #495057;
"""

EMPTY_DISPLAY_STYLE = """
color: #adb5bd;
font-size: 18px;
font-weight: light;
"""

# Info text style
INFO_TEXT_STYLE = """
color: #6c757d;
font-size: 12px;
"""

def get_image_type_filter():
    """Return file filter for image selection dialog"""
    return "Image Files (*.jpg *.jpeg *.png *.bmp);;All Files (*)"

def get_video_type_filter():
    """Return file filter for video selection dialog"""
    return "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"

def get_weights_type_filter():
    """Return file filter for weights selection dialog"""
    return "Weights Files (*.pt,*.onnx,*.engine);;All Files (*)"