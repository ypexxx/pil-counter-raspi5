from flask import Flask, render_template, jsonify, request, send_file
from utils.pipeline import run_pipeline
from hardware.gpio_control import wait_button_press, cleanup_gpio, led_on, led_off, led_atas_on, led_atas_off
import threading
import logging
import time
import os
import shutil
import cv2
import atexit
from datetime import datetime
from config import IMAGE_PATH, TEMP_DIR, CAMERA_INDEX

# ============================================
# Setup logging
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Nyalakan Led
logger.info("1. Turning LED ON...")
led_on()
led_atas_on()

# ============================================
# Global State
# ============================================
class AppState:
    def __init__(self):
        self.status = "idle"  # idle, processing, done, error
        self.count = 0
        self.image_path = None
        self.result_image_path = None
        self.timestamp = None
        self.detection_time = None
        self.error_message = None
        self._lock = threading.Lock()
    
    def update(self, **kwargs):
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
    
    def get(self):
        with self._lock:
            return {
                "status": self.status,
                "count": self.count,
                "image_path": self.image_path,
                "result_image_path": self.result_image_path,
                "timestamp": self.timestamp,
                "detection_time": self.detection_time,
                "error_message": self.error_message
            }
    
    def reset(self):
        with self._lock:
            self.status = "idle"
            self.count = 0
            self.image_path = None
            self.result_image_path = None
            self.timestamp = None
            self.detection_time = None
            self.error_message = None

state = AppState()

# Frame buffer untuk video streaming
latest_frame = None
frame_lock = threading.Lock()
camera_active = True

# ============================================
# Helper Functions
# ============================================
def safe_remove_file(filepath):
    """Safely delete a file without crashing if it's locked or missing."""
    if filepath and os.path.exists(filepath):
        try:
            os.remove(filepath)
            logger.debug(f"Removed file: {filepath}")
        except Exception as e:
            logger.warning(f"Could not remove file {filepath}: {e}")

def ensure_directories():
    """Ensure all required directories exist"""
    os.makedirs(TEMP_DIR, exist_ok=True)
    result_dir = os.path.join(TEMP_DIR, "result")
    os.makedirs(result_dir, exist_ok=True)
    logger.info(f"Directories created: {TEMP_DIR}, {result_dir}")

# ============================================
# Pipeline Functions
# ============================================
def process_detection():
    """
    Main detection process:
    1. Capture frame from camera
    2. Run YOLO pipeline
    3. Update state with results
    """
    global latest_frame
    
    try:
        # Check if already processing
        current_status = state.get()["status"]
        if current_status == "processing":
            logger.warning("Pipeline already running. Ignoring duplicate request.")
            return

        # 1. Get latest frame from camera
        with frame_lock:
            if latest_frame is None:
                logger.error("No frame available from camera")
                state.update(
                    status="error",
                    error_message="Kamera tidak merespon"
                )
                return
            frame_to_save = latest_frame.copy()

        # 2. Update status to processing
        state.update(
            status="processing",
            timestamp=datetime.now().isoformat()
        )
        logger.info("Detection started")

        # 3. Ensure directories exist
        ensure_directories()
        
        # 4. Save frame to temp directory
        cv2.imwrite(IMAGE_PATH, frame_to_save)
        logger.info(f"Frame saved to {IMAGE_PATH}")
        state.update(image_path=IMAGE_PATH)
        
        # 5. Clear previous result
        result_path = os.path.join(TEMP_DIR, "result", "result.jpg")
        safe_remove_file(result_path)
        
        # 6. Run YOLO pipeline (capture_image() sudah di-handle di dalam)
        logger.info("Running YOLO detection pipeline...")
        start_time = time.time()
        
        # Jalankan pipeline (tanpa capture ulang)
        count = run_pipeline()  # Pipeline sudah handle capture dan deteksi
        detection_time = time.time() - start_time
        
        # 7. Update state with results
        if os.path.exists(result_path):
            state.update(
                status="done",
                count=count,
                result_image_path=result_path,
                detection_time=detection_time,
                timestamp=datetime.now().isoformat()
            )
        else:
            # Jika tidak ada result image, gunakan original
            state.update(
                status="done",
                count=count,
                result_image_path=IMAGE_PATH,
                detection_time=detection_time,
                timestamp=datetime.now().isoformat()
            )
        
        logger.info(f"Detection completed: {count} pills found in {detection_time:.2f}s")
        
        # 8. Schedule auto-reset after showing result
        def auto_reset():
            time.sleep(10)  # Tampilkan hasil selama 10 detik
            if state.get()["status"] == "done":
                logger.info("Auto-resetting state after 10s")
                state.reset()
        
        reset_thread = threading.Thread(target=auto_reset, daemon=True)
        reset_thread.start()
        
    except Exception as e:
        logger.error(f"Detection error: {e}", exc_info=True)
        state.update(
            status="error",
            error_message=str(e),
            count=0
        )
        # Clean up hardware
        try:
            from hardware.gpio_control import led_off, led_atas_off, vibrator_off
            led_off()
            led_atas_off()
            vibrator_off()
        except:
            pass

# ============================================
# Video Streaming Functions
# ============================================
def generate_video_stream():
    """Generator untuk MJPEG video streaming"""
    global latest_frame, camera_active
    
    camera = None
    try:
        # Buka kamera
        camera = cv2.VideoCapture(CAMERA_INDEX)
        if not camera.isOpened():
            logger.error("Failed to open camera for streaming")
            camera_active = False
            # Kirim placeholder jika kamera gagal
            placeholder = create_placeholder_image()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
            return
        
        camera_active = True
        logger.info("Camera streaming started")
        
        # Set camera properties
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        
        TARGET_FPS = 10
        INTERVAL = 1.0 / TARGET_FPS
        frame_skip = 0
        
        while True:
            start_time = time.time()
            
            # Cek status
            current_status = state.get()["status"]
            
            # Jika processing, pause stream (kirim frame terakhir)
            if current_status == "processing":
                # Kirim frame terakhir yang tersimpan
                with frame_lock:
                    if latest_frame is not None:
                        ret, buffer = cv2.imencode('.jpg', latest_frame)
                        if ret:
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.1)
                continue
            
            # Baca frame dari kamera
            ret, frame = camera.read()
            if not ret or frame is None:
                logger.warning("Failed to read frame from camera")
                time.sleep(0.1)
                continue
            
            # Simpan frame terakhir
            with frame_lock:
                latest_frame = frame.copy()
            
            # Kompres ke JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # FPS limiting
            elapsed = time.time() - start_time
            if elapsed < INTERVAL:
                time.sleep(INTERVAL - elapsed)
                
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        camera_active = False
    finally:
        if camera is not None:
            camera.release()
            logger.info("Camera released")

def create_placeholder_image():
    """Create placeholder image when camera is not available"""
    import numpy as np
    # Buat gambar hitam dengan teks
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, "Camera Not Available", (120, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    _, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()

# ============================================
# Button Listener Thread
# ============================================
def button_listener():
    """Background thread untuk mendeteksi tombol fisik"""
    logger.info("Button listener started")
    while True:
        try:
            # Wait for button press (non-blocking with timeout)
            if wait_button_press(timeout=0.5):
                logger.info("Button pressed - starting detection")
                # Start detection in separate thread
                thread = threading.Thread(target=process_detection, daemon=True)
                thread.start()
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"Button listener error: {e}")
            time.sleep(1)

# ============================================
# Routes
# ============================================
@app.route("/video_feed")
def video_feed():
    """Route untuk MJPEG video streaming"""
    from flask import Response
    return Response(generate_video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/")
def index():
    """Halaman utama dengan live preview"""
    ensure_directories()
    # Reset state jika idle
    if state.get()["status"] not in ["processing", "done"]:
        state.reset()
    return render_template("index.html", page="home")

@app.route("/start")
def start_page():
    """Halaman processing"""
    # Jika status idle, redirect ke home
    if state.get()["status"] == "idle":
        return render_template("start.html", 
                             image_exists=False,
                             image_path=None,
                             timestamp=datetime.now().isoformat())
    
    # Jika status done, redirect ke result
    if state.get()["status"] == "done":
        return render_template("result.html")
    
    # Jika processing, tampilkan halaman start
    data = state.get()
    return render_template("start.html",
                         image_exists=os.path.exists(IMAGE_PATH),
                         image_path=IMAGE_PATH if os.path.exists(IMAGE_PATH) else None,
                         timestamp=datetime.now().isoformat())

@app.route("/result")
def result_page():
    """Halaman hasil"""
    # Jika status idle atau processing, redirect
    current_status = state.get()["status"]
    if current_status == "idle":
        return render_template("result.html")
    elif current_status == "processing":
        return render_template("start.html",
                             image_exists=os.path.exists(IMAGE_PATH),
                             image_path=IMAGE_PATH if os.path.exists(IMAGE_PATH) else None,
                             timestamp=datetime.now().isoformat())
    
    return render_template("result.html")

# ============================================
# API Endpoints
# ============================================
@app.route("/api/status")
def api_status():
    """API untuk mendapatkan status sistem"""
    data = state.get()
    
    # Check if result image exists
    if data["status"] == "done":
        result_path = os.path.join(TEMP_DIR, "result", "result.jpg")
        data["result_image_exists"] = os.path.exists(result_path)
    
    # Check if original image exists
    data["image_exists"] = os.path.exists(IMAGE_PATH)
    
    return jsonify(data)

@app.route("/api/start", methods=["POST"])
def api_start():
    """API untuk memulai deteksi"""
    current_status = state.get()["status"]
    
    if current_status == "processing":
        return jsonify({
            "success": False,
            "message": "Deteksi sedang berjalan"
        })
    
    # Clear previous result
    result_path = os.path.join(TEMP_DIR, "result", "result.jpg")
    safe_remove_file(result_path)
    
    # Start detection in background
    thread = threading.Thread(target=process_detection, daemon=True)
    thread.start()
    
    return jsonify({
        "success": True,
        "status": "processing"
    })

@app.route("/api/reset", methods=["POST"])
def api_reset():
    """API untuk mereset state"""
    state.reset()
    logger.info("State reset by API")
    return jsonify({"success": True})

@app.route("/api/camera_status")
def api_camera_status():
    """API untuk cek status kamera"""
    return jsonify({
        "active": camera_active,
        "frame_available": latest_frame is not None
    })

@app.route("/image/<path:filename>")
def serve_image(filename):
    """Serve image files from temp directory"""
    try:
        # Security: only serve from temp directory
        full_path = os.path.join(TEMP_DIR, filename)
        if not os.path.exists(full_path):
            # Try with result subdirectory
            full_path = os.path.join(TEMP_DIR, "result", filename)
            if not os.path.exists(full_path):
                logger.warning(f"Image not found: {filename}")
                return "Image not found", 404
        
        return send_file(full_path)
    except Exception as e:
        logger.error(f"Image serve error: {e}")
        return "Image not found", 404


# ============================================
# Cleanup on exit
# ============================================
def cleanup():
    """Clean up resources on exit"""
    logger.info("Cleaning up resources...")
    try:
        cleanup_gpio()
    except:
        pass
    logger.info("Cleanup complete")

atexit.register(cleanup)

# ============================================
# Main Application
# ============================================
if __name__ == "__main__":
    # Ensure directories exist
    ensure_directories()
    
    # Test camera on startup
    try:
        from vision.camera import test_camera
        if test_camera():
            logger.info("Camera test passed")
        else:
            logger.warning("Camera test failed - check camera connection")
    except Exception as e:
        logger.error(f"Camera test error: {e}")
    
    # Start button listener thread
    button_thread = threading.Thread(target=button_listener, daemon=True)
    button_thread.start()
    logger.info("Button listener thread started")
    
    # Start Flask app
    logger.info("Starting Flask server on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)