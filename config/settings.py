"""
Configuration settings for the YOLOv5 Detector application.
Contains default values and argument parsing functionality.
"""
import argparse

def get_default_args():
    """
    Define and return the default configuration arguments.
    
    Returns:
        argparse.Namespace: Default arguments
    """
    parser = argparse.ArgumentParser(description='YOLOv5 Object Detection')
    parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov5s.pt',
                        help='model path(s)')
    parser.add_argument('--source', type=str, default='data/images',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default='data/coco128.yaml',
                        help='(optional) dataset.yaml path')
    parser.add_argument('--img-size', nargs='+', type=int, default=640,
                        help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                        help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                        help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000,
                        help='maximum detections per image')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='show results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true',
                        help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true',
                        help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--visualize', action='store_true',
                        help='visualize features')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    parser.add_argument('--project', default='runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int,
                        help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true',
                        help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true',
                        help='hide confidences')
    
    return parser.parse_args([])  # Parse with empty list to get defaults

# Application constants
APP_NAME = "YOLO Detector"
APP_VERSION = "1.0"