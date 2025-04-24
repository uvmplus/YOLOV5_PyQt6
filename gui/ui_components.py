"""
Reusable UI components for the YOLOv5 Detector application.
"""
from PyQt6 import QtCore, QtWidgets, QtGui
from PyQt6.QtWidgets import QLabel, QTextBrowser, QPushButton, QGroupBox, QVBoxLayout, QHBoxLayout
from gui.styles import *

class ControlPanel(QtWidgets.QWidget):
    """Left control panel with buttons"""
    
    def __init__(self, parent=None):
        """Initialize the control panel"""
        super().__init__(parent)
        self.setMaximumWidth(240)
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the control panel UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)
        
        # Title
        self.title_label = QLabel("YOLO Detector")
        self.title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet(TITLE_LABEL_STYLE)
        layout.addWidget(self.title_label)
        
        # Model setup group
        self.model_group = QGroupBox("Model Setup")
        model_layout = QVBoxLayout(self.model_group)
        
        self.button_weights = QPushButton("Select Weights")
        self.button_weights.setObjectName("pushButton_weights")
        self.button_weights.setIcon(QtGui.QIcon.fromTheme("document-open"))
        
        self.button_init = QPushButton("Initialize Model")
        self.button_init.setObjectName("pushButton_init")
        self.button_init.setIcon(QtGui.QIcon.fromTheme("system-run"))
        
        model_layout.addWidget(self.button_weights)
        model_layout.addWidget(self.button_init)
        layout.addWidget(self.model_group)
        
        # Detection group
        self.detect_group = QGroupBox("Detection")
        detect_layout = QVBoxLayout(self.detect_group)
        
        self.button_img = QPushButton("Image Detection")
        self.button_img.setObjectName("pushButton_img")
        self.button_img.setIcon(QtGui.QIcon.fromTheme("image-x-generic"))
        
        self.button_video = QPushButton("Video Detection")
        self.button_video.setObjectName("pushButton_video")
        self.button_video.setIcon(QtGui.QIcon.fromTheme("video-x-generic"))
        
        self.button_camera = QPushButton("Camera Detection")
        self.button_camera.setObjectName("pushButton_camera")
        self.button_camera.setIcon(QtGui.QIcon.fromTheme("camera-web"))
        
        detect_layout.addWidget(self.button_img)
        detect_layout.addWidget(self.button_video)
        detect_layout.addWidget(self.button_camera)
        layout.addWidget(self.detect_group)
        
        # 在ControlPanel类的setup_ui方法中修改视频控制部分:
        # Video controls group
        self.control_group = QGroupBox("Video Controls")
        control_layout = QVBoxLayout(self.control_group)

        # 直接将按钮添加到垂直布局
        self.button_stop = QPushButton("Pause")
        self.button_stop.setObjectName("pushButton_stop")
        self.button_stop.setIcon(QtGui.QIcon.fromTheme("media-playback-pause"))
        self.button_stop.setEnabled(False)

        self.button_finish = QPushButton("Stop")
        self.button_finish.setObjectName("pushButton_finish")
        self.button_finish.setIcon(QtGui.QIcon.fromTheme("media-playback-stop"))
        self.button_finish.setEnabled(False)

        # 垂直添加按钮（上下排列）
        control_layout.addWidget(self.button_stop)
        control_layout.addWidget(self.button_finish)
        layout.addWidget(self.control_group)
        
        # Add stretch to push info to bottom
        layout.addStretch()
        
        # App info at bottom
        self.app_info = QLabel("YOLOv5 Object Detection")
        self.app_info.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.app_info.setStyleSheet(INFO_TEXT_STYLE)
        layout.addWidget(self.app_info)


class DisplayPanel(QtWidgets.QWidget):
    """Right panel with display area and status info"""
    
    def __init__(self, parent=None):
        """Initialize the display panel"""
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the display panel UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(15)
        
        # Display frame
        self.display_frame = QtWidgets.QFrame()
        self.display_frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.display_frame.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.display_frame.setStyleSheet(DISPLAY_FRAME_STYLE)
        display_layout = QVBoxLayout(self.display_frame)
        
        self.display_label = QLabel()
        self.display_label.setObjectName("label")
        self.display_label.setText("No image loaded")
        self.display_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.display_label.setStyleSheet(EMPTY_DISPLAY_STYLE)
        self.display_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, 
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        self.display_label.setMinimumHeight(400)
        display_layout.addWidget(self.display_label)
        layout.addWidget(self.display_frame)
        
        # Status frame
        self.status_frame = QtWidgets.QFrame()
        self.status_frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.status_frame.setStyleSheet(STATUS_FRAME_STYLE)
        status_layout = QVBoxLayout(self.status_frame)
        
        self.status_header = QLabel("Detection Status")
        self.status_header.setStyleSheet(STATUS_LABEL_STYLE)
        
        self.status_display = QTextBrowser()
        self.status_display.setObjectName("status_display")
        self.status_display.setFixedHeight(120)
        self.status_display.setPlaceholderText("Detection information will appear here...")
        
        status_layout.addWidget(self.status_header)
        status_layout.addWidget(self.status_display)
        layout.addWidget(self.status_frame)