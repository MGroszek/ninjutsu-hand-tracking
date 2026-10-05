import socket
import time
import cv2
import mediapipe as mp
import math
from mediapipe.tasks.python import BaseOptions, vision
from pythonosc.udp_client import SimpleUDPClient

OFFSET = 1.2        # jak wysoko nad dłonią (w "dłoniach")
REF_Hand = 0.13     # rozmiar dłoni przy normalnej odległości
GROW_TIME = 0.35    # NOWE: czas pojawiania się kuli w sekundach
SHRINK_TIME = 0.3  # NOWE: czas znikania kuli w sekundach


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


def is_fist(lm, aspect):
    # palec zgięty = czubek bliżej nadgarstka niż środkowy staw
    folded = 0
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        if dist2d(lm[tip], lm[0], aspect) < dist2d(lm[pip], lm[0], aspect):
            folded += 1
    return folded == 4


# --- połączenie z Unrealem ---
ip = local_ip()
print("Wysyłam do:", ip)
client = SimpleUDPClient(ip, 8000)

# --- detektor dłoni ---
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
sx = None
sy = None
sh = None
t_on = 0.0          # NOWE: moment włączenia kuli
t_off = 10.0        # NOWE: moment wyłączenia kuli
fist_frames = 0        # NOWE: licznik klatek z pięścią

try:
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

            # pięść - liczymy, ile klatek z rzędu trwa
            fist = is_fist(lm, aspect)
            if fist:
                fist_frames += 1
            else:
                fist_frames = 0

            # szczypnięcie = WŁĄCZ (tylko gdy dłoń NIE jest pięścią)
            d = dist2d(lm[4], lm[8], aspect) / hand
            if not pinched and d < 0.25 and not fist:
                pinched = True
                if not active and time.time() - t_off > 0.5:
                    active = True
                    t_on = time.time()
            elif pinched and d > 0.40:
                pinched = False

            # pięść przez 5 klatek = WYŁĄCZ
            if active and fist_frames >= 5:
                active = False
                t_off = time.time()

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

            # odległość kuli od kamery (rozmiar)
            dist = 500 * REF_Hand / sh
            dist = max(150, min(900, dist))

            now = time.time()
            if active:
                g = min(1.0, (now - t_on) / GROW_TIME)
                # pojawianie: szybko, potem hamuje
                g = 1 - (1 - g) ** 3
                visible = True
            else:
                p = min(1.0, (now - t_off) / SHRINK_TIME)
                g = (1 - p) ** 2              # znikanie: maleje coraz szybciej
                visible = p < 1.0             # widoczna, dopóki animacja trwa
            dist = min(4500, dist / max(g, 0.1))
            # pozycja 3D w Unrealu
            ux = dist
            uy = (sx - 0.5) * 2 * dist
            uz = 5500 + (0.5 - sy) * 1.125 * dist

            client.send_message(
                "/hand", [sx, sy, 1.0 if active else 0.0, ux, uy, uz,
                          1.0 if visible else 0.0])

except KeyboardInterrupt:          # Ctrl+C = spokojne zakończenie
    print("Koniec.")
finally:                           # wykona się ZAWSZE, nawet po błędzie
    cap.release()
    detector.close()

    # .venv/bin/python hand_osc.py
