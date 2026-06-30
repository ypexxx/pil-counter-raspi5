import time
import logging
from hardware.gpio_control import led_on, led_off, vibrator_on, vibrator_off
from vision.camera import capture_image
from vision.detector import count_pills
from config import IMAGE_PATH

logger = logging.getLogger(__name__)

def run_pipeline():
    """
    Main pipeline untuk deteksi obat:
    1. Nyalakan LED indikator
    2. Aktifkan vibrator
    3. Ambil gambar dari direktori /temp
    4. Jalankan deteksi YOLOv8
    5. Matikan LED
    Returns: jumlah obat terdeteksi
    """
    count = 0
    
    try:
        logger.info("=" * 50)
        logger.info("STARTING DETECTION PIPELINE")
        logger.info("=" * 50)

        # 2. Nyalakan vibrator
        vibrator_on()
        time.sleep(1)
        vibrator_off()
        
        # 3. Ambil gambar dari direktori /temp
        logger.info("3. Get image...")
        image_path = IMAGE_PATH
        
        # success, image_path = capture_image(IMAGE_PATH)
        # if not success:
        #     logger.error("Failed to capture image")
        #     led_off()
        #     return 0
        # logger.info(f"Image captured: {image_path}")
        
        # 4. Jalankan deteksi
        logger.info("4. Running YOLO detection...")
        start_time = time.time()
        count, result_path = count_pills(image_path)
        detection_time = time.time() - start_time
        
        logger.info(f"Detection completed in {detection_time:.2f}s")
        logger.info(f"Count: {count} pills detected")
        
        # 5. Matikan LED
        logger.info("5. Turning LED OFF...")
        led_off()
        
        logger.info("=" * 50)
        logger.info(f"PIPELINE COMPLETED: {count} pills found")
        logger.info("=" * 50)
        
        return count
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        # Clean up hardware
        try:
            led_off()
            vibrator_off()
        except:
            pass
        return 0