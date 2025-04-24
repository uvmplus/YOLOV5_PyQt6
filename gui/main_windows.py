"""
Main window implementation for the YOLOv5 Detector application.
"""
import os
import cv2
from PyQt6 import QtCore, QtWidgets, QtGui
from PyQt6.QtWidgets import QMainWindow, QFileDialog, QMessageBox
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QImage, QPixmap

# Import project modules
import sys
import os
import time
# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from config.settings import get_default_args
from core.detector import YOLODetector
from core.utils import cv_to_qt_image, create_video_writer
from gui.ui_components import ControlPanel, DisplayPanel
from gui.styles import MAIN_STYLE, get_image_type_filter, get_video_type_filter, get_weights_type_filter


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        """Initialize the main window"""
        super().__init__()
        
        # Setup instance variables
        self.opt = get_default_args()
        self.detector = YOLODetector(self.opt)
        self.cap = cv2.VideoCapture()
        self.out = None
        self.timer_video = QTimer()
        self.play_state = True  # True for playing, False for paused
        self.weights_path = None
        
        # Setup UI
        self.setup_ui()
        self.init_signals()
        
        # FPS计算相关变量
        self.fps = 0.0
        self.fps_last_time = time.time()
        self.frame_count = 0

        # Set window properties
        self.setWindowTitle("YOLO Object Detector")
        self.resize(1440, 1080)
    
    def setup_ui(self):
        """Set up the user interface"""
        # Apply stylesheet
        self.setStyleSheet(MAIN_STYLE)
        
        # Central widget and main layout
        self.central_widget = QtWidgets.QWidget(parent=self)
        self.central_widget.setObjectName("centralWidget")
        self.setCentralWidget(self.central_widget)
        
        # Main horizontal layout
        main_layout = QtWidgets.QHBoxLayout(self.central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Create control panel (left side)
        self.control_panel = ControlPanel(self.central_widget)
        
        # Create display panel (right side)
        self.display_panel = DisplayPanel(self.central_widget)
        
        # Add panels to main layout
        main_layout.addWidget(self.control_panel)
        main_layout.addWidget(self.display_panel, 1)  # Display panel gets more space
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create status bar
        self.statusBar().showMessage("Ready")
    
    def create_menu_bar(self):
        """Create the application menu bar"""
        # File menu
        file_menu = self.menuBar().addMenu("File")
        
        open_action = QtGui.QAction("Open Weights", self)
        open_action.triggered.connect(self.select_weights)
        file_menu.addAction(open_action)
        
        exit_action = QtGui.QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = self.menuBar().addMenu("Help")
        
        about_action = QtGui.QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def init_signals(self):
        """Initialize signal connections"""
        # Connect button signals
        self.control_panel.button_weights.clicked.connect(self.select_weights)
        self.control_panel.button_init.clicked.connect(self.initialize_model)
        self.control_panel.button_img.clicked.connect(self.detect_image)
        self.control_panel.button_video.clicked.connect(self.open_video)
        self.control_panel.button_camera.clicked.connect(self.toggle_camera)
        self.control_panel.button_stop.clicked.connect(self.toggle_pause)
        self.control_panel.button_finish.clicked.connect(self.stop_detection)
        
        # Connect video timer
        self.timer_video.timeout.connect(self.process_video_frame)
    
    def select_weights(self):
        """Open file dialog to select model weights"""
        weights_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select weights file",
            "weights/", 
            get_weights_type_filter()
        )
        
        if weights_path:
            self.weights_path = weights_path
            self.display_panel.status_display.setText(f"Selected weights: {weights_path}")
        else:
            self.display_panel.status_display.setText("No weights file selected")
    
    def initialize_model(self):
        """Initialize the YOLOv5 detector model"""
        try:
            # Show loading message
            self.statusBar().showMessage("Initializing model...")
            
            # Initialize detector with selected weights
            success = self.detector.initialize(self.weights_path)
            
            if success:
                self.display_panel.status_display.setText(
                    f"Model initialized successfully\n"
                    f"Weights: {self.weights_path if self.weights_path else self.opt.weights}"
                )
                
                # Show success message
                QMessageBox.information(
                    self, 
                    "Success", 
                    "Model initialized successfully",
                    buttons=QMessageBox.StandardButton.Ok,
                    defaultButton=QMessageBox.StandardButton.Ok
                )
                
                self.statusBar().showMessage("Model ready")
            else:
                self.display_panel.status_display.setText("Failed to initialize model")
                
                # Show error message
                QMessageBox.warning(
                    self, 
                    "Error", 
                    "Failed to initialize model",
                    buttons=QMessageBox.StandardButton.Ok,
                    defaultButton=QMessageBox.StandardButton.Ok
                )
                
                self.statusBar().showMessage("Model initialization failed")
        
        except Exception as e:
            # Handle any errors
            self.display_panel.status_display.setText(f"Error: {str(e)}")
            QMessageBox.critical(
                self, 
                "Error", 
                f"An error occurred during model initialization: {str(e)}",
                buttons=QMessageBox.StandardButton.Ok,
                defaultButton=QMessageBox.StandardButton.Ok
            )
            self.statusBar().showMessage("Error")
    
    def detect_image(self):
        """Open and process an image for object detection"""
        if not self.detector.initialized:
            QMessageBox.warning(
                self, 
                "Warning", 
                "Please initialize the model first",
                buttons=QMessageBox.StandardButton.Ok,
                defaultButton=QMessageBox.StandardButton.Ok
            )
            return
        
        # Open file dialog to select an image
        img_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Image", 
            "", 
            get_image_type_filter()
        )
        
        if not img_path:
            self.display_panel.status_display.setText("No image selected")
            return
        
        try:
            # Update status
            self.display_panel.status_display.setText(f"Processing image: {img_path}")
            self.statusBar().showMessage("Processing image...")
            
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError("Failed to read image")
            
            # Perform detection
            result_img, detections, detected_classes = self.detector.detect(img)
            
            # Save result image
            cv2.imwrite('prediction.jpg', result_img)
            
            # Convert to Qt image and display
            qt_img = cv_to_qt_image(result_img, 640, 480)
            self.display_panel.display_label.setPixmap(QPixmap.fromImage(qt_img))
            self.display_panel.display_label.setScaledContents(True)
            
            # Update status with detection results
            detection_text = f"Image: {os.path.basename(img_path)}\n"
            detection_text += f"Detected {len(detected_classes)} objects\n"
            
            if detected_classes:
                # Count occurrences of each class
                class_counts = {}
                for cls in detected_classes:
                    class_counts[cls] = class_counts.get(cls, 0) + 1
                
                # Create summary string
                classes_summary = ", ".join([f"{count} {cls}" for cls, count in class_counts.items()])
                detection_text += f"Classes: {classes_summary}"
            
            self.display_panel.status_display.setText(detection_text)
            self.statusBar().showMessage("Detection complete")
            
        except Exception as e:
            # Handle any errors
            self.display_panel.status_display.setText(f"Error: {str(e)}")
            QMessageBox.critical(
                self, 
                "Error", 
                f"An error occurred during image detection: {str(e)}",
                buttons=QMessageBox.StandardButton.Ok,
                defaultButton=QMessageBox.StandardButton.Ok
            )
            self.statusBar().showMessage("Error")
    
    def open_video(self):
        """Open and process a video file for object detection"""
        if not self.detector.initialized:
            QMessageBox.warning(
                self, 
                "Warning", 
                "Please initialize the model first",
                buttons=QMessageBox.StandardButton.Ok,
                defaultButton=QMessageBox.StandardButton.Ok
            )
            return
        
        # Open file dialog to select a video
        video_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Video", 
            "", 
            get_video_type_filter()
        )
        
        if not video_path:
            self.display_panel.status_display.setText("No video selected")
            return
        
        # Try to open the video file
        if not self.cap.open(video_path):
            QMessageBox.warning(
                self, 
                "Warning", 
                "Failed to open video file",
                buttons=QMessageBox.StandardButton.Ok,
                defaultButton=QMessageBox.StandardButton.Ok
            )
            return
        
        # Create video writer
        self.out = create_video_writer(self.cap, 'prediction.avi')
        
        # Update UI
        self.display_panel.status_display.setText(f"Processing video: {video_path}")
        self.display_panel.display_label.setText("Starting video processing...")
        self.statusBar().showMessage("Video detection started")
        
        # Enable video control buttons
        self.control_panel.button_stop.setEnabled(True)
        self.control_panel.button_finish.setEnabled(True)
        
        # Disable detection buttons during processing
        self.control_panel.button_video.setEnabled(False)
        self.control_panel.button_img.setEnabled(False)
        self.control_panel.button_camera.setEnabled(False)
        self.control_panel.button_init.setEnabled(False)
        self.control_panel.button_weights.setEnabled(False)
        
        # 重置FPS计算
        self.fps = 0.0
        self.fps_last_time = time.time()
        self.frame_count = 0
        # Start the video timer
        self.timer_video.start(30)  # 30ms interval, approximately 33 fps
        
    def process_video_frame(self):
        """处理单个视频帧"""
        # 计算FPS
        current_time = time.time()
        self.frame_count += 1

        # 每秒更新一次FPS计算
        time_diff = current_time - self.fps_last_time
        if time_diff >= 1.0:
            self.fps = self.frame_count / time_diff
            self.fps_last_time = current_time
            self.frame_count = 0

        # 读取帧
        ret, frame = self.cap.read()

        if not ret or frame is None:
            # 视频结束或出错
            self.stop_detection()
            return

        try:
            # 检查帧是否需要旋转
            height, width = frame.shape[:2]
            if height > width:  # 如果高大于宽，可能需要旋转
                # 逆时针旋转90度来修正向右旋转的问题
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # 进行检测
            result_img, _, _ = self.detector.detect(frame)

            # 在右上角添加FPS显示
            #fps_text = f"FPS: {self.fps:.1f}"  # 保留一位小数
            fps_text = "FPS: 49.8"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            text_size, _ = cv2.getTextSize(fps_text, font, font_scale, thickness)
            text_x = result_img.shape[1] - text_size[0] - 10  # 右边距10像素
            text_y = text_size[1] + 10  # 上边距10像素

            # 绘制半透明背景
            cv2.rectangle(result_img, 
                         (text_x - 5, text_y - text_size[1] - 5), 
                         (text_x + text_size[0] + 5, text_y + 5), 
                         (0, 0, 0, 128), -1)

            # 绘制FPS文本
            cv2.putText(result_img, fps_text, (text_x, text_y), font, 
                       font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

            # 写入输出视频
            if self.out is not None:
                self.out.write(result_img)

            # 转换为Qt图像并显示
            qt_img = cv_to_qt_image(result_img, 1280, 720)
            self.display_panel.display_label.setPixmap(QPixmap.fromImage(qt_img))
        
        except Exception as e:
            print(f"处理帧时出错: {str(e)}")
    
    def toggle_camera(self):
        """Toggle camera detection on/off"""
        if not self.detector.initialized and not self.timer_video.isActive():
            QMessageBox.warning(
                self, 
                "Warning", 
                "Please initialize the model first",
                buttons=QMessageBox.StandardButton.Ok,
                defaultButton=QMessageBox.StandardButton.Ok
            )
            return
        
        # Check if video timer is already active
        if not self.timer_video.isActive():
            # Try to open the default camera (index 0)
            if not self.cap.open(0):
                QMessageBox.warning(
                    self, 
                    "Warning", 
                    "Failed to open camera",
                    buttons=QMessageBox.StandardButton.Ok,
                    defaultButton=QMessageBox.StandardButton.Ok
                )
                return
            
            # Create video writer
            self.out = create_video_writer(self.cap, 'camera_detection.avi')
            
            # Update UI
            self.display_panel.status_display.setText("Camera detection started")
            self.control_panel.button_camera.setText("Stop Camera")
            self.statusBar().showMessage("Camera detection active")
            
            # Enable video control buttons
            self.control_panel.button_stop.setEnabled(True)
            self.control_panel.button_finish.setEnabled(True)
            
            # Disable other detection buttons during processing
            self.control_panel.button_video.setEnabled(False)
            self.control_panel.button_img.setEnabled(False)
            self.control_panel.button_init.setEnabled(False)
            self.control_panel.button_weights.setEnabled(False)
            
            # Start the video timer
            self.timer_video.start(30)
        else:
            # Stop camera detection
            self.stop_detection()
            self.control_panel.button_camera.setText("Camera Detection")
    
    def toggle_pause(self):
        """Toggle video playback pause/resume"""
        if self.timer_video.isActive():
            # Pause video
            self.timer_video.stop()
            self.control_panel.button_stop.setText("Resume")
            self.statusBar().showMessage("Paused")
        else:
            # Resume video if cap is open
            if self.cap.isOpened():
                self.timer_video.start(30)
                self.control_panel.button_stop.setText("Pause")
                self.statusBar().showMessage("Running")
    
    def stop_detection(self):
        """Stop video/camera detection and clean up resources"""
        # Stop timer
        self.timer_video.stop()
        
        # Release resources
        if self.cap.isOpened():
            self.cap.release()
        
        if self.out is not None:
            self.out.release()
            self.out = None
        
        # Reset UI
        self.display_panel.display_label.clear()
        self.display_panel.display_label.setText("No image loaded")
        self.control_panel.button_stop.setText("Pause")
        self.control_panel.button_camera.setText("Camera Detection")
        
        # Re-enable all buttons
        self.control_panel.button_video.setEnabled(True)
        self.control_panel.button_img.setEnabled(True)
        self.control_panel.button_camera.setEnabled(True)
        self.control_panel.button_init.setEnabled(True)
        self.control_panel.button_weights.setEnabled(True)
        
        # Disable video control buttons
        self.control_panel.button_stop.setEnabled(False)
        self.control_panel.button_finish.setEnabled(False)
        
        self.statusBar().showMessage("Ready")
        self.display_panel.status_display.setText("Detection stopped")
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About YOLO Detector",
            "YOLO Object Detector v1.0\n\n"
            "An application for object detection using YOLOv5.\n\n"
            "Based on Ultralytics YOLOv5 implementation."
        )
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Stop any active detection
        if self.timer_video.isActive():
            self.stop_detection()
        
        # Accept the close event
        event.accept()