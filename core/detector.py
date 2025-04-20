"""
YOLOv5 detector implementation.
Handles model loading, inference, and detection processing.
"""
import random
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.plots import plot_one_box
from utils.torch_utils import select_device

class YOLODetector:
    """YOLOv5 model detector class"""
    
    def __init__(self, opt):
        """
        Initialize the YOLOv5 detector.
        
        Args:
            opt (argparse.Namespace): Configuration options
        """
        self.opt = opt
        self.device = None
        self.model = None
        self.names = None
        self.colors = None
        self.imgsz = None
        self.half = False
        self.initialized = False
    
    def initialize(self, weights_path=None):
        """
        Initialize the detector with model weights.
        
        Args:
            weights_path (str, optional): Path to model weights. 
                                         If None, use opt.weights. Defaults to None.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            # Use provided weights path or from options
            weights = weights_path if weights_path else self.opt.weights
            
            # Select device
            self.device = select_device(self.opt.device)
            self.half = self.device.type != 'cpu'  # Half precision only on CUDA
            
            # Enable cuDNN benchmarking for improved performance
            cudnn.benchmark = True
            
            # Load model
            self.model = attempt_load(
                weights=weights, device=self.device, inplace=True, fuse=True)
            
            # Get model stride and check image size
            stride = int(self.model.stride.max())
            self.imgsz = check_img_size(self.opt.img_size, s=stride)
            
            # Convert to half precision if using CUDA
            if self.half:
                self.model.half()
            
            # Get class names and generate random colors
            self.names = self.model.module.names if hasattr(
                self.model, 'module') else self.model.names
            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
            
            # Mark as initialized
            self.initialized = True
            return True
            
        except Exception as e:
            print(f"Error initializing YOLOv5 detector: {str(e)}")
            return False
    
    def preprocess_image(self, img):
        """
        Preprocess an image for detection.
        
        Args:
            img (numpy.ndarray): Input image in BGR format
        
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Resize and pad image
        img = letterbox(img, new_shape=self.imgsz)[0]
        
        # Convert BGR to RGB, to 3xHxW
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        
        # Convert to torch tensor
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        
        # Add batch dimension if necessary
        if img.ndim == 3:
            img = img.unsqueeze(0)
        
        return img
    
    def detect(self, img):
        """
        Perform detection on an image.
        
        Args:
            img (numpy.ndarray): Input image in BGR format
        
        Returns:
            tuple: (processed_image, detections_list, class_names_list)
        """
        if not self.initialized:
            raise RuntimeError("Detector not initialized. Call initialize() first.")
        
        # Store original image for drawing
        original_img = img.copy()
        detected_classes = []
        
        # Preprocess image
        processed_img = self.preprocess_image(img)
        
        # Perform inference
        with torch.no_grad():
            # Forward pass
            pred = self.model(processed_img, augment=self.opt.augment)[0]
            
            # Apply NMS
            pred = non_max_suppression(
                pred, 
                self.opt.conf_thres, 
                self.opt.iou_thres,
                classes=self.opt.classes,
                agnostic=self.opt.agnostic_nms
            )
            
            # Process detections
            for i, det in enumerate(pred):
                if det is not None and len(det):
                    # Rescale boxes from img_size to original image size
                    det[:, :4] = scale_boxes(
                        processed_img.shape[2:], det[:, :4], original_img.shape).round()
                    
                    # Process each detection
                    for *xyxy, conf, cls in reversed(det):
                        class_name = self.names[int(cls)]
                        detected_classes.append(class_name)
                        
                        # Add bounding box to image
                        label = f'{class_name} {conf:.2f}'
                        plot_one_box(
                            xyxy, 
                            original_img, 
                            label=label, 
                            color=self.colors[int(cls)],
                            line_thickness=self.opt.line_thickness
                        )
        
        return original_img, pred, detected_classes