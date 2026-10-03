import socket
import time
import cv2
import mediapipe as mp
import math
from mediapipe.tasks.python import BaseOptions, vision
from pythonosc.udp_client import SimpleUDPClient

OFFSET = 1.2
REF_Hand = 0.13


def local_ip():
    """Sprawdza aktualny adres tego Maca w sieci."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    return ip


def palm_flatness(lm):
    # dwa wektory wzdłuż dłoni: nadgarstek→wskazujący, nadgarstek→mały
    a = (lm[5].x - lm[0].x, lm[5].y - lm[0].y, lm[5].z - lm[0].z)
    b = (lm[17].x - lm[0].x, lm[17].y - lm[0].y, lm[17].z - lm[0].z)
    # iloczyn wektorowy = strzałka prostopadła do dłoni
    nx = a[1] * b[2] - a[2] * b[1]
    ny = a[2] * b[0] - a[0] * b[2]
    nz = a[0] * b[1] - a[1] * b[0]
    length = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
    # 0 = dłoń pionowo, 1 = dłoń płasko
    return abs(ny) / length


def dist2d(a, b, aspect):
    # odległość na obrazie, y przeliczone na jednostki x
    return math.hypot(a.x - b.x, (a.y - b.y) * aspect)


# --- połączenie z Unrealem ---
ip = local_ip()
print("Wysyłam do:", ip)
client = SimpleUDPClient(ip, 8000)

# --- detektor dłoni (tak jak w menu.py) ---
option = vision.HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task",
        delegate=BaseOptions.Delegate.CPU,
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
)
detector = vision.HandLandmarker.create_from_options(option)
cap = cv2.VideoCapture(0)
failures = 0
sent = 0

active = False
pinched = False
sh = None
try:
    sx = None
    sy = None
    while True:
        ok, frame = cap.read()
        if not ok:
            failures += 1
            if failures > 30:
                print("Cannot read frame")
                break
            continue
        failures = 0

        frame = cv2.flip(frame, 1)  # lustro - tak samo jak w Unrealu
        aspect = frame.shape[0] / frame.shape[1]   # wysokość / szerokość
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        picture = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect_for_video(picture, int(time.time() * 1000))

        if result.hand_landmarks:
            # punkty dłoni
            lm = result.hand_landmarks[0]
            length = dist2d(lm[0], lm[9], aspect)
            width = dist2d(lm[5], lm[17], aspect)
            hand = max(length, width * 1.3)

            # szczypnięcie
            d = dist2d(lm[4], lm[8], aspect) / hand
            if not pinched and d < 0.25:
                pinched = True
                active = not active
            elif pinched and d > 0.40:
                pinched = False

            # środek dłoni + przesunięcie nad dłoń
            x = (lm[0].x + lm[9].x) / 2
            y = (lm[0].y + lm[9].y) / 2
            flat = palm_flatness(lm)
            y = y - (hand / aspect) * OFFSET * flat

            # wygładzanie pozycji
            if sx is None:
                sx, sy = x, y
            else:
                sx += (x - sx) * 0.5
                sy += (y - sy) * 0.5

            # wygładzanie rozmiaru dłoni
            if sh is None:
                sh = hand
            else:
                sh += (hand - sh) * 0.3

            # pozycja 3D w Unrealu
            dist = 500 * REF_Hand / sh
            dist = max(150, min(900, dist))
            ux = dist
            uy = (sx - 0.5) * 2 * dist
            uz = 500 + (0.5 - sy) * 1.125 * dist

            client.send_message(
                "/hand", [sx, sy, 1.0 if active else 0.0, ux, uy, uz])
            sent += 1
            if sent % 15 == 0:
                print(f"hand={sh:.3f} dist={dist:.0f}")
except KeyboardInterrupt:          # Ctrl+C = spokojne zakończenie, bez czerwonego błędu
    print("Koniec.")
finally:                           # wykona się ZAWSZE, nawet po błędzie
    cap.release()

    detector.close()
