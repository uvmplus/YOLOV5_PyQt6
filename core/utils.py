"""
Utility functions for the YOLOv5 Detector application.
"""
import os
import sys
import cv2
from PyQt6.QtGui import QImage, QPixmap

def get_project_root():
    """
    Get the project root directory.
    
    Returns:
        str: Path to project root
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)

def add_yolov5_to_path():
    """
    Add YOLOv5 root directory to Python path.
    """
    yolov5_root = get_project_root()
    if yolov5_root not in sys.path:
        sys.path.append(yolov5_root)
        print(f"Added YOLOv5 root to path: {yolov5_root}")

def cv_to_qt_image(cv_img, target_width=None, target_height=None):
    """
    Convert OpenCV image to QImage for display.
    
    Args:
        cv_img (numpy.ndarray): OpenCV image (BGR format)
        target_width (int, optional): Target width for resizing
        target_height (int, optional): Target height for resizing
    
    Returns:
        QImage: Qt compatible image
    """
    # Convert BGR to RGB
    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    
    # Resize if dimensions specified
    if target_width and target_height:
        rgb_img = cv2.resize(rgb_img, (target_width, target_height), 
                            interpolation=cv2.INTER_AREA)
    
    # Create QImage from numpy array
    height, width, channels = rgb_img.shape
    bytes_per_line = channels * width
    qt_img = QImage(rgb_img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
    
    return qt_img

def create_video_writer(cap, output_path='prediction.avi'):
    """
    Create a video writer based on input video properties.
    
    Args:
        cap (cv2.VideoCapture): Video capture object
        output_path (str, optional): Output video path. Defaults to 'prediction.avi'.
    
    Returns:
        cv2.VideoWriter: Video writer object
    """
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Default to 20 fps if unable to determine
    if fps <= 0:
        fps = 20
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    return writer