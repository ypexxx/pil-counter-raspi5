import cv2
import os
import time
import logging
from config import CAMERA_INDEX, IMAGE_PATH

logger = logging.getLogger(__name__)

def capture_image(image_path=None):
    """
    Capture image from camera and save to path
    Returns: (success: bool, image_path: str)
    """
    try:
        # Use default path if not specified
        if image_path is None:
            image_path = IMAGE_PATH
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        
        # Open camera
        logger.info(f"Opening camera {CAMERA_INDEX}...")
        cap = cv2.VideoCapture(CAMERA_INDEX)
        
        if not cap.isOpened():
            logger.error(f"Failed to open camera {CAMERA_INDEX}")
            return False, None
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Wait for camera to warm up
        time.sleep(0.2)
        
        # Capture frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            logger.error("Failed to read frame from camera")
            return False, None
        
        # Save image
        cv2.imwrite(image_path, frame)
        logger.info(f"Image captured and saved to {image_path}")
        
        return True, image_path
        
    except Exception as e:
        logger.error(f"Camera capture error: {e}")
        return False, None

def test_camera():
    """Test if camera is working"""
    try:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            logger.error("Camera test failed: Cannot open camera")
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            logger.info("Camera test successful")
            return True
        else:
            logger.error("Camera test failed: Cannot read frame")
            return False
            
    except Exception as e:
        logger.error(f"Camera test error: {e}")
        return False