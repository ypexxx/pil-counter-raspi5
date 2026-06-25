from gpiozero import Button, LED, OutputDevice
from config import BUTTON_PIN, LED_PIN, VIBRATOR_PIN
import time
import logging

logger = logging.getLogger(__name__)

# Inisialisasi hardware
button = Button(BUTTON_PIN, pull_up=True)
led = LED(LED_PIN)
vibrator = OutputDevice(VIBRATOR_PIN)

def led_on():
    """Turn LED on"""
    try:
        led.on()
        logger.debug("LED ON")
    except Exception as e:
        logger.error(f"LED on error: {e}")

def led_off():
    """Turn LED off"""
    try:
        led.off()
        logger.debug("LED OFF")
    except Exception as e:
        logger.error(f"LED off error: {e}")

def vibrator_on():
    """Turn vibrator on"""
    try:
        vibrator.on()
        logger.debug("Vibrator ON")
    except Exception as e:
        logger.error(f"Vibrator on error: {e}")

def vibrator_off():
    """Turn vibrator off"""
    try:
        vibrator.off()
        logger.debug("Vibrator OFF")
    except Exception as e:
        logger.error(f"Vibrator off error: {e}")

def wait_button_press(timeout=None):
    """
    Wait for button press with optional timeout
    Returns: True if button pressed, False if timeout
    """
    try:
        if timeout:
            # Wait with timeout
            start = time.time()
            while time.time() - start < timeout:
                if button.is_pressed:
                    # Wait for release
                    while button.is_pressed:
                        time.sleep(0.01)
                    logger.info("Button pressed")
                    return True
                time.sleep(0.01)
            logger.debug("Button wait timeout")
            return False
        else:
            # Wait indefinitely
            button.wait_for_press()
            # Wait for release
            while button.is_pressed:
                time.sleep(0.01)
            logger.info("Button pressed")
            return True
    except Exception as e:
        logger.error(f"Button wait error: {e}")
        return False

def button_is_pressed():
    """Check if button is currently pressed (non-blocking)"""
    try:
        return button.is_pressed
    except Exception as e:
        logger.error(f"Button check error: {e}")
        return False

def cleanup_gpio():
    """Clean up GPIO resources"""
    try:
        led.off()
        vibrator.off()
        logger.info("GPIO cleaned up")
    except Exception as e:
        logger.error(f"GPIO cleanup error: {e}")