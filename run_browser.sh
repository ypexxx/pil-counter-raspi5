#!/bin/bash
# Jeda waktu 5 detik agar sistem & server app.py siap sepenuhnya
sleep 5

# Buka Firefox dalam mode Kiosk (Fullscreen penuh tanpa UI browser)
# Ganti port 5000 sesuai dengan port yang digunakan oleh app.py Anda (misal: 8000 atau 5000)
firefox --kiosk http://127.0.0.1:5000