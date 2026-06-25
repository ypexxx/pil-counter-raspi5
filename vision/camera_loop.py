import cv2
import time
import logging
from config import IMAGE_PATH

logger = logging.getLogger(__name__)

def camera_loop():
    """Continuously capture image every 2 seconds"""

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logger.error("Camera failed to open")
        return

    while True:
        try:
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(IMAGE_PATH, frame)
            else:
                logger.warning("Failed to capture frame")

            time.sleep(3)

        except Exception as e:
            logger.error(f"Camera loop error: {e}")
            time.sleep(3)