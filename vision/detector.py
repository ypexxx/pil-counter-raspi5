from ultralytics import YOLO
import cv2
import os
import logging
from config import MODEL_PATH, RESULT_DIR

logger = logging.getLogger(__name__)

# Global model instance (lazy loading)
_model = None

def load_model():
    """Load YOLO model (lazy loading)"""
    global _model
    if _model is None:
        try:
            if not os.path.exists(MODEL_PATH):
                logger.error(f"Model not found at {MODEL_PATH}")
                raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
            
            logger.info(f"Loading YOLO model from {MODEL_PATH}...")
            _model = YOLO(MODEL_PATH)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    return _model

def count_pills(image_path, conf_threshold=0.5, iou_threshold=0.45):
    """
    Detect and count pills using YOLOv8
    
    Returns:
        count: Number of pills detected
        result_path: Path to annotated image
    """
    try:
        # Load model
        model = load_model()
        
        # Check if image exists
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return 0, None
        
        # Load image for saving result
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return 0, None
        
        # Run inference
        logger.info(f"Running YOLO inference on {image_path}...")
        results = model(
            image_path, 
            conf=conf_threshold, 
            iou=iou_threshold,
            verbose=False
        )
        
        # Count objects
        count = 0
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                count = len(boxes)
        
        logger.info(f"Detected {count} pills")
        
        # Save annotated image
        os.makedirs(RESULT_DIR, exist_ok=True)
        result_path = os.path.join(RESULT_DIR, "result.jpg")
        
        if count > 0 and results and len(results) > 0:
            # Annotate image with bounding boxes
            annotated_img = results[0].plot()
            cv2.imwrite(result_path, annotated_img)
            logger.info(f"Annotated image saved to {result_path}")
        else:
            # Save original image if no detection
            cv2.imwrite(result_path, img)
            logger.info("No pills detected, saved original image")
        
        return count, result_path
        
    except Exception as e:
        logger.error(f"Error in count_pills: {e}", exc_info=True)
        return 0, None

def get_detection_details(image_path):
    """Get detailed detection information"""
    try:
        model = load_model()
        
        if not os.path.exists(image_path):
            return None
        
        results = model(image_path, conf=0.5, iou=0.45, verbose=False)
        
        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                class_names = model.names if hasattr(model, 'names') else {}
                
                for box in boxes:
                    detection = {
                        'class': int(box.cls[0]) if box.cls is not None else -1,
                        'confidence': float(box.conf[0]) if box.conf is not None else 0.0,
                    }
                    if box.xyxy is not None:
                        detection['bbox'] = box.xyxy[0].tolist()
                    if detection['class'] in class_names:
                        detection['class_name'] = class_names[detection['class']]
                    
                    detections.append(detection)
        
        return {
            'count': len(detections),
            'detections': detections
        }
        
    except Exception as e:
        logger.error(f"Error getting detection details: {e}")
        return None