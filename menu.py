import math
import random
import time
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------- MENU KOŁOWE ----------
TECHNIQUES = ["Rasengan", "Chidori", "Katon",
              "Kage Bunshin"]  # góra, prawo, dół, lewo
MENU_RADIUS = 220   # promień koła w pikselach
DEAD_ZONE = 40      # środek koła = nic nie wybrane
PINCH_ON = 0.25   # poniżej = szczypnięcie się zaczyna
PINCH_OFF = 0.40  # powyżej = szczypnięcie puszczone
CLONE_OFFSET = 0.28  # odległość klonów od Ciebie (część szerokości obrazu)
SMOKE_TIME = 0.8     # ile sekund trwa dym przy pojawieniu się klonów


def get_selected(center, finger):
    """Zwraca numer wycinka (0-3), na który wskazuje palec, albo None."""
    dx = finger[0] - center[0]
    dy = finger[1] - center[1]
    if math.hypot(dx, dy) < DEAD_ZONE:
        return None
    angle = math.degrees(math.atan2(dy, dx))  # -180..180, 0 = w prawo
    return int(((angle + 135) % 360) // 90)


TECH_COLORS = {                      # kolory BGR każdej techniki
    "Rasengan": (255, 160, 40),      # niebieski
    "Chidori": (255, 220, 150),      # jasny błękit
    "Katon": (0, 120, 255),          # pomarańczowy
    "Kage Bunshin": (210, 210, 210),  # biało-szary
}
# płynne podświetlenie każdego wycinka (0..1)
menu_hover = [0.0, 0.0, 0.0, 0.0]


def draw_text_outline(img, text, pos, scale, color, thickness=2):
    """Napis z czarną obwódką = czytelny na każdym tle."""
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0),
                thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_DUPLEX, scale, color,
                thickness, cv2.LINE_AA)


def draw_menu(frame, center, selected):
    """Półprzezroczyste koło z kolorowymi wycinkami i płynnym podświetleniem."""
    h, w, _ = frame.shape
    pad = MENU_RADIUS + 30
    x1, y1 = max(center[0] - pad, 0), max(center[1] - pad, 0)
    x2, y2 = min(center[0] + pad, w), min(center[1] + pad, h)
    roi = frame[y1:y2, x1:x2]
    c = (center[0] - x1, center[1] - y1)

    # 1. płynne podświetlenie: wartość "dojeżdża" do celu, zamiast skakać
    for i in range(4):
        target = 1.0 if i == selected else 0.0
        menu_hover[i] += (target - menu_hover[i]) * 0.3

    # 2. kolorowe wycinki na kopii (z przerwami między nimi)
    overlay = roi.copy()
    for i, name in enumerate(TECHNIQUES):
        hv = menu_hover[i]
        start = -135 + i * 90 + 3
        end = start + 84
        # wybrany lekko "wyskakuje"
        r = int(MENU_RADIUS + 18 * hv)
        color = TECH_COLORS[name]
        k = 0.35 + 0.55 * hv                           # jasność: ciemny -> kolorowy
        fill = tuple(int(30 * (1 - k) + ch * k) for ch in color)
        cv2.ellipse(overlay, c, (r, r), 0, start, end, fill, -1, cv2.LINE_AA)

    # 3. dziura w środku (widać kamerę), potem półprzezroczystość
    cv2.circle(overlay, c, DEAD_ZONE, (0, 0, 0), -1)
    hole = np.zeros(roi.shape[:2], np.uint8)
    cv2.circle(hole, c, DEAD_ZONE, 255, -1)
    overlay[hole > 0] = roi[hole > 0]
    roi[:] = cv2.addWeighted(overlay, 0.7, roi, 0.3, 0)

    # 4. obwódki i napisy (ostre, na wierzchu)
    for i, name in enumerate(TECHNIQUES):
        hv = menu_hover[i]
        start = -135 + i * 90 + 3
        end = start + 84
        r = int(MENU_RADIUS + 18 * hv)
        color = TECH_COLORS[name]
        cv2.ellipse(roi, c, (r, r), 0, start, end,
                    (255, 255, 255), 1, cv2.LINE_AA)
        if hv > 0.05:  # świecąca krawędź wybranego wycinka
            cv2.ellipse(roi, c, (r, r), 0, start, end,
                        color, int(1 + 5 * hv), cv2.LINE_AA)

        mid = math.radians(start + 42)
        scale = 0.6 + 0.2 * hv
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, scale, 2)
        tx = int(c[0] + math.cos(mid) * r * 0.66) - tw // 2
        ty = int(c[1] + math.sin(mid) * r * 0.66) + th // 2
        draw_text_outline(roi, name, (tx, ty), scale, (255, 255, 255))

    cv2.circle(roi, c, DEAD_ZONE, (255, 255, 255), 1, cv2.LINE_AA)


RASENGAN_RINGS = [  # 3 pierścienie energii, każdy pochylony inaczej (kąty w radianach)
    (0.3, 0.0), (1.2, 1.0), (-0.9, 2.1),
]
# rozmiar "płótna" efektu w pikselach (stały = zawsze szybko)
RS = 320
SPHERE_RES = 160    # kula liczona w jeszcze mniejszej rozdzielczości
PAD = 1.9           # płótno sięga 1.9 promienia kuli od środka
_sphere = None      # rzeczy, które NIE zmieniają się w czasie, liczymy tylko raz


def _prepare_sphere():
    """Liczy raz: kształt kuli, oświetlenie, poświatę. Potem tylko używamy."""
    xs = np.linspace(-PAD, PAD, SPHERE_RES)
    nx, ny = np.meshgrid(xs, xs)
    d2 = nx * nx + ny * ny
    inside = d2 < 1
    nz = np.sqrt(np.clip(1 - d2, 0, 1))                 # "wysokość" kuli -> 3D
    static = (0.25 * np.clip(-0.2 * nx - 0.5 * ny + nz, 0, 1)  # światło z lewej-góry
              + 0.9 * (1 - nz) ** 2                     # jasne brzegi
              + 0.9 * nz ** 6) * inside                 # jasny środek
    static += np.exp(-(np.sqrt(d2) - 1) * 3.5) * (~inside) * 0.9  # poświata
    return {
        "static": static.astype(np.float32),
        "lon": np.arctan2(nx, nz + 1e-6).astype(np.float32),
        "lat": np.arcsin(np.clip(ny, -1, 1)).astype(np.float32),
        "nz": (nz * inside).astype(np.float32),
    }


def draw_rasengan(frame, center, size, t):
    """Rasengan "3D" liczony na małym płótnie, potem powiększony (szybko!)."""
    global _sphere
    if _sphere is None:
        _sphere = _prepare_sphere()
    s = _sphere

    # 1. kula: zmienia się tylko wirująca tekstura
    swirl = np.sin(7 * (s["lon"] + t * 6) + 5 *
                   np.sin(2 * s["lat"] + t * 3)) * 0.5 + 0.5
    total = s["static"] + 0.8 * swirl ** 3 * s["nz"]
    base = np.array([255, 140, 30], np.float32) / 255      # niebieski (BGR)
    col = total[:, :, None] * base * 1.3 + \
        np.clip(total - 0.8, 0, 1)[:, :, None]
    layer = cv2.resize((np.clip(col, 0, 1) * 255).astype(np.uint8), (RS, RS))

    # 2. komety energii na orbitach 3D (z tyłu chowają się za kulą)
    r = RS / 2 / PAD                                    # promień kuli na płótnie
    c = RS // 2
    front = np.zeros_like(layer)
    back = np.zeros_like(layer)
    for k, (tilt, phase) in enumerate(RASENGAN_RINGS):
        for comet in range(3):
            head = t * (5 + k) + phase + comet * 2 * math.pi / 3
            prev = None
            for j in range(12):                        # 12 punktów ogona
                a = head - j * 0.1
                X, Z = math.cos(a) * 1.3, math.sin(a) * 1.3
                Y, Z = -Z * math.sin(tilt), Z * math.cos(tilt)
                persp = 1 + Z * 0.25                   # bliżej = większe
                p = (int(c + X * r * persp), int(c + Y * r * persp))
                if prev is not None:
                    thick = max(int(r * 0.08 * persp * (1 - j / 12)), 1)
                    if Z > 0:
                        cv2.line(front, prev, p, (255, 235, 200),
                                 thick, cv2.LINE_AA)
                    elif X * X + Y * Y > 1:
                        cv2.line(back, prev, p, (200, 120, 40),
                                 thick, cv2.LINE_AA)
                prev = p
    front = cv2.add(front, cv2.GaussianBlur(front, (0, 0), 3))

    # 3. płótno -> właściwy rozmiar na ekranie (+ lekkie pulsowanie)
    side = int(2 * PAD * size * (1 + 0.05 * math.sin(t * 10)))
    if side < 8:
        return
    shade = np.zeros((RS, RS), np.uint8)
    cv2.circle(shade, (c, c), int(r), 110, -1, cv2.LINE_AA)
    parts = [cv2.resize(img, (side, side))
             for img in (back, layer, front, shade)]

    h, w, _ = frame.shape
    x, y = center[0] - side // 2, center[1] - side // 2
    x1, y1, x2, y2 = max(x, 0), max(y, 0), min(x + side, w), min(y + side, h)
    if x2 <= x1 or y2 <= y1:
        return
    cut = (slice(y1 - y, y2 - y), slice(x1 - x, x2 - x))
    back, layer, front, shade = (p[cut] for p in parts)

    roi = frame[y1:y2, x1:x2]
    roi[:] = cv2.add(roi, back)
    dark = cv2.cvtColor(255 - shade, cv2.COLOR_GRAY2BGR)  # kula zasłania tło
    roi[:] = cv2.multiply(roi, dark, scale=1 / 255)
    roi[:] = cv2.add(roi, layer)
    roi[:] = cv2.add(roi, front)


def random_dir():
    """Losowy kierunek w 3D (wektor o długości 1)."""
    v = np.random.normal(size=3)
    return v / (np.linalg.norm(v) + 1e-6)


def bolt_3d(start, direction, length, steps):
    """Błyskawica w 3D: idziemy krokami i przy każdym lekko losowo skręcamy."""
    pts = [np.array(start, np.float32)]
    d = np.array(direction, np.float32)
    for _ in range(steps):
        d = d + random_dir() * 0.45
        d /= np.linalg.norm(d) + 1e-6
        pts.append(pts[-1] + d * (length / steps))
    return pts


def draw_chidori(frame, center, size):
    """Chidori "3D": rozgałęzione błyskawice w przestrzeni. Bliższe są grubsze
    i jaśniejsze, dalsze cieńsze i ciemniejsze; rysujemy od najdalszych."""
    r = max(size, 5)
    h, w, _ = frame.shape
    pad = int(r * 3)
    x1, y1 = max(center[0] - pad, 0), max(center[1] - pad, 0)
    x2, y2 = min(center[0] + pad, w), min(center[1] + pad, h)
    if x2 <= x1 or y2 <= y1:
        return
    cx, cy = center[0] - x1, center[1] - y1

    # 1. budujemy błyskawice (w jednostkach promienia r) + odgałęzienia
    bolts = []
    for _ in range(random.randint(10, 14)):
        main = bolt_3d(random_dir() * 0.3, random_dir(),
                       random.uniform(1.6, 2.8), 9)
        bolts.append(main)
        for _ in range(random.randint(1, 2)):
            i = random.randint(2, len(main) - 2)
            bolts.append(bolt_3d(main[i], random_dir(),
                         random.uniform(0.4, 0.9), 4))

    # 2. rozbijamy na odcinki i sortujemy od najdalszego (z) do najbliższego
    segs = []
    for pts in bolts:
        for a, b in zip(pts, pts[1:]):
            segs.append(((a[2] + b[2]) / 2, a, b))
    segs.sort(key=lambda s: s[0])

    def project(p):
        # bliżej kamery = dalej od środka
        persp = 1 + p[2] * 0.25
        return (int(cx + p[0] * r * persp), int(cy + p[1] * r * persp))

    glow = np.zeros((y2 - y1, x2 - x1, 3), np.uint8)
    sharp = np.zeros_like(glow)
    flicker = random.uniform(0.75, 1.0)              # cała błyskawica migocze
    for z, a, b in segs:
        f = min(max((z + 1.5) / 3, 0), 1) * flicker  # 0 = daleko, 1 = blisko
        pa, pb = project(a), project(b)
        g = 0.5 + 0.5 * f                            # dalsze też trochę świecą
        cv2.line(glow, pa, pb, (int(255 * g), int(170 * g), int(70 * g)),
                 int(6 + 14 * f), cv2.LINE_AA)
        cv2.line(sharp, pa, pb, (255, int(200 + 55 * f), int(150 + 105 * f)),
                 max(int(1 + 2.5 * f), 1), cv2.LINE_AA)

    # 3. świecące "serce" w dłoni
    cv2.circle(glow, (cx, cy), int(r * 0.9), (255, 140, 50), -1, cv2.LINE_AA)
    cv2.circle(sharp, (cx, cy), int(r * 0.28),
               (255, 255, 255), -1, cv2.LINE_AA)

    # 4. poświata (liczona na małej kopii) + ostre błyskawice na wierzchu
    small = cv2.resize(glow, None, fx=0.25, fy=0.25)
    small = cv2.GaussianBlur(small, (0, 0), max(r * 0.05, 1))
    glow = cv2.resize(small, (x2 - x1, y2 - y1))
    roi = frame[y1:y2, x1:x2]
    roi[:] = cv2.add(roi, glow)
    roi[:] = cv2.add(roi, glow // 2)
    # wąska aura wokół każdej iskry
    halo = cv2.resize(sharp, None, fx=0.5, fy=0.5)
    halo = cv2.resize(cv2.GaussianBlur(halo, (0, 0), 1.5), (x2 - x1, y2 - y1))
    roi[:] = cv2.add(roi, cv2.multiply(halo, (1.0, 0.8, 0.5, 0)))
    roi[:] = cv2.add(roi, sharp)


def draw_katon(frame, center, size, particles):
    """Katon "3D": płomienie z głębią (bliższe iskry większe i jaśniejsze),
    gorący środek, a nad ogniem ciemny dym, który daje objętość.
    particles = lista, która PAMIĘTA cząsteczki między klatkami."""
    h, w, _ = frame.shape
    size = max(size, 5)

    # 1. nowe płomienie w dłoni, każdy z głębokością z (-1 daleko, +1 blisko)
    for _ in range(20):
        particles.append({
            "kind": "fire",
            "x": center[0] + random.uniform(-0.35, 0.35) * size,
            "y": center[1] + random.uniform(-0.2, 0.2) * size,
            "z": random.uniform(-1, 1),
            "vx": random.uniform(-0.03, 0.03) * size,
            "vy": random.uniform(-0.17, -0.08) * size,
            "life": 1.0,
            "r": random.uniform(0.12, 0.24) * size,
        })

    # 2. ruch; wygasły płomień czasem zamienia się w dym
    new_smoke = []
    for p in particles:
        p["vx"] += random.uniform(-0.02, 0.02) * size   # płomień "tańczy"
        p["x"] += p["vx"]
        p["y"] += p["vy"]
        if p["kind"] == "fire":
            p["life"] -= random.uniform(0.035, 0.07)
            if p["life"] <= 0 and random.random() < 0.25:
                new_smoke.append({"kind": "smoke", "x": p["x"], "y": p["y"], "z": p["z"],
                                  "vx": p["vx"] * 0.5, "vy": p["vy"] * 0.5,
                                  "life": 1.0, "r": p["r"] * 1.5})
        else:
            p["life"] -= 0.03
            p["r"] *= 1.04                              # dym się rozszerza
    particles[:] = [p for p in particles if p["life"] > 0] + new_smoke
    if not particles:
        return

    # 3. pracujemy tylko w prostokącie, gdzie jest ogień (a nie na całym ekranie)
    margin = size * 1.2
    x1 = int(max(min(p["x"] for p in particles) - margin, 0))
    x2 = int(min(max(p["x"] for p in particles) + margin, w))
    y1 = int(max(min(p["y"] for p in particles) - margin, 0))
    y2 = int(min(max(p["y"] for p in particles) + margin, h))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return

    # 4. rysujemy na płótnie w 1/3 rozdzielczości, od najdalszych
    S = 3
    sw, sh = (x2 - x1) // S + 1, (y2 - y1) // S + 1
    fire = np.zeros((sh, sw, 3), np.uint8)
    core = np.zeros((sh, sw, 3), np.uint8)
    smoke = np.zeros((sh, sw), np.uint8)
    for p in sorted(particles, key=lambda q: q["z"]):
        persp = 1 + 0.35 * p["z"]                        # bliżej = większe
        pos = (int((p["x"] - x1) / S), int((p["y"] - y1) / S))
        life = p["life"]
        if p["kind"] == "smoke":
            cv2.circle(smoke, pos, int(
                p["r"] * persp / S), int(110 * life), -1)
            continue
        bright = 0.6 + 0.4 * (p["z"] + 1) / 2             # bliżej = jaśniej
        color = (0, int(200 * life ** 1.5 * bright), int(255 * bright))
        rad = max(int(p["r"] * (0.4 + life) * persp / S), 1)
        cv2.circle(fire, pos, rad, color, -1)
        if life > 0.6:                                   # młode = biało-żółty środek
            cv2.circle(core, pos, max(rad // 2, 1), (120, 230, 255), -1)

    # 5. rozmycia = miękkość; dym przyciemnia, ogień rozjaśnia
    k = size / S
    smoke = cv2.GaussianBlur(smoke, (0, 0), max(k * 0.15, 1))
    light = cv2.add(cv2.GaussianBlur(fire, (0, 0), max(k * 0.2, 1)),
                    cv2.GaussianBlur(fire, (0, 0), max(k * 0.06, 1)))
    light = cv2.add(light, cv2.GaussianBlur(core, (0, 0), max(k * 0.06, 1)))
    dark = cv2.cvtColor(255 - smoke, cv2.COLOR_GRAY2BGR)
    dark = cv2.resize(dark, (x2 - x1, y2 - y1))
    light = cv2.resize(light, (x2 - x1, y2 - y1))

    roi = frame[y1:y2, x1:x2]
    roi[:] = cv2.multiply(roi, dark, scale=1 / 255)
    roi[:] = cv2.add(roi, light)


def place(img, dx, scale, anchor_x, h, w):
    """Przesuwa obraz o dx i zmniejsza go (scale), "stojąc" na dole kadru.
    Mniejszy = dalej od kamery -> wrażenie głębi."""
    m = np.float32([[scale, 0, (1 - scale) * anchor_x + dx],
                    [0, scale, (1 - scale) * h]])
    return cv2.warpAffine(img, m, (w, h))


def draw_clones(frame, mask, elapsed):
    """Kage Bunshin "3D": 2 klony stoją dalej za Tobą: są mniejsze,
    ciemniejsze i lekko rozmyte (jak w obiektywie). Na start - dym."""
    h, w, _ = frame.shape
    appear = min(elapsed / SMOKE_TIME, 1.0)          # 0 -> 1 w czasie dymu
    offset = int(w * CLONE_OFFSET)

    # środek osoby (liczony na małej masce, żeby było szybko)
    small = cv2.resize(mask, (80, 45))
    ys, xs = np.nonzero(small > 0.5)
    if len(xs) == 0:
        return
    cx = int(xs.mean() * w / 80)
    cy = int(ys.mean() * h / 45)

    original = frame.copy()
    # "dalej" = lekko nieostre
    far = cv2.GaussianBlur(original, (0, 0), 2.5)
    far = cv2.convertScaleAbs(far, alpha=0.75)        # i trochę ciemniejsze
    result = frame.copy()

    for dx in (-offset, offset):
        clone = place(far, dx, 0.85, cx, h, w)
        weight = place(mask, dx, 0.85, cx, h, w) * appear
        result = cv2.blendLinear(clone, result, weight, 1 - weight)

    frame[:] = cv2.blendLinear(
        original, result, mask, 1 - mask)  # Ty na wierzchu

    if appear < 1.0:
        sy = int(h - (h - cy) * 0.85)
        draw_smoke(frame, [(int(cx * 0.85 + 0.15 * cx) - offset, sy),
                           (int(cx * 0.85 + 0.15 * cx) + offset, sy)], appear, h * 0.3)


def draw_smoke(frame, centers, progress, size):
    """Biały kłąb dymu, który rośnie i znika (progress 0 -> 1)."""
    layer = np.zeros_like(frame)
    for cx, cy in centers:
        for _ in range(30):
            a = random.uniform(0, 2 * math.pi)
            d = random.uniform(0, size * (0.3 + progress))
            x = int(cx + math.cos(a) * d)
            y = int(cy + math.sin(a) * d * 1.4)  # dym wyższy niż szerszy
            r = int(size * random.uniform(0.15, 0.3) * (0.6 + progress))
            cv2.circle(layer, (x, y), r, (230, 230, 230), -1)
    small = cv2.resize(layer, None, fx=0.25, fy=0.25)
    small = cv2.GaussianBlur(small, (0, 0), max(size * 0.03, 1))
    smoke = cv2.resize(small, (frame.shape[1], frame.shape[0]))
    strength = 1.0 - progress  # dym z czasem znika
    frame[:] = cv2.add(frame, (smoke * strength).astype(np.uint8))


FONT_PATHS = [  # pierwsza czcionka, która istnieje na Macu, wygrywa
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]
title_cache = {}  # gotowe obrazki napisów, żeby nie rysować ich co klatkę


def load_font(size):
    for path in FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def make_title(text, color):
    """Rysuje napis RAZ (Pillow): kolor techniki, czarny obrys i poświata.
    Zwraca obrazek BGR i przezroczystość (0..1)."""
    font = load_font(110)
    stroke = 7
    box = font.getbbox(text, stroke_width=stroke)
    margin = 40
    img = Image.new("RGBA", (box[2] - box[0] + 2 *
                    margin, box[3] - box[1] + 2 * margin))
    draw = ImageDraw.Draw(img)
    rgb = (color[2], color[1], color[0])  # BGR -> RGB
    draw.text((margin - box[0], margin - box[1]), text, font=font,
              fill=rgb + (255,), stroke_width=stroke, stroke_fill=(0, 0, 0, 255))

    arr = np.array(img).astype(np.float32) / 255
    text_bgr = arr[:, :, 2::-1]              # RGB -> BGR
    alpha = arr[:, :, 3:4]
    glow = cv2.GaussianBlur(alpha, (0, 0), 14)[:, :, None] * 0.9
    color_img = np.ones_like(text_bgr) * (np.array(color, np.float32) / 255)

    bgr = alpha * text_bgr + (1 - alpha) * color_img  # napis na tle poświaty
    a = alpha + (1 - alpha) * glow
    return bgr, a


def draw_title(frame, name, elapsed):
    """Napis techniki: wlatuje duży i "uderza" w miejsce, potem zostaje."""
    if name not in title_cache:
        title_cache[name] = make_title(name.upper() + "!", TECH_COLORS[name])
    bgr, alpha = title_cache[name]

    pop = max(0.0, 1 - elapsed / 0.25)   # 1 -> 0 w ciągu 0.25 s
    scale = 1 + 0.8 * pop * pop          # najpierw 1.8x, potem normalny
    fade = min(elapsed / 0.12, 1.0)      # szybkie pojawienie się
    if scale != 1:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale)
        alpha = cv2.resize(alpha, None, fx=scale, fy=scale)[:, :, None]
    alpha = alpha * fade

    h, w, _ = frame.shape
    th, tw = alpha.shape[:2]
    x, y = (w - tw) // 2, 130 - th // 2
    # przycinamy, jeśli napis wystaje poza ekran
    fx1, fy1 = max(x, 0), max(y, 0)
    fx2, fy2 = min(x + tw, w), min(y + th, h)
    if fx2 <= fx1 or fy2 <= fy1:
        return
    part = (slice(fy1 - y, fy2 - y), slice(fx1 - x, fx2 - x))
    roi = frame[fy1:fy2, fx1:fx2].astype(np.float32) / 255
    out = alpha[part] * bgr[part] + (1 - alpha[part]) * roi
    frame[fy1:fy2, fx1:fx2] = (out * 255).astype(np.uint8)


def flash_and_shake(frame, color, elapsed):
    """Błysk w kolorze techniki + wstrząs obrazu przez pierwsze ułamki sekundy."""
    if elapsed < 0.35:  # wstrząs słabnie z czasem
        amp = 18 * (1 - elapsed / 0.35)
        dx, dy = random.uniform(-amp, amp), random.uniform(-amp, amp)
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        h, w, _ = frame.shape
        frame[:] = cv2.warpAffine(
            frame, m, (w, h), borderMode=cv2.BORDER_REFLECT)
    if elapsed < 0.3:   # błysk gaśnie z czasem
        s = 0.6 * (1 - elapsed / 0.3)
        flash = np.full_like(frame, color)
        frame[:] = cv2.addWeighted(frame, 1 - s, flash, s, 0)


# ---------- DETEKTOR DŁONI ----------
option = vision.HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task",
        delegate=BaseOptions.Delegate.CPU,
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
)
detector = vision.HandLandmarker.create_from_options(option)

# ---------- SEGMENTACJA OSOBY (do klonów) ----------
seg_option = vision.ImageSegmenterOptions(
    base_options=BaseOptions(
        model_asset_path="selfie_segmenter.tflite",
        delegate=BaseOptions.Delegate.CPU,
    ),
    running_mode=vision.RunningMode.VIDEO,
    output_confidence_masks=True,
)
segmenter = vision.ImageSegmenter.create_from_options(seg_option)

cap = cv2.VideoCapture(0)
# mniejszy obraz = mniej pracy w każdej klatce
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
fps = 30.0                                # licznik klatek na sekundę
last_time = time.time()
failures = 0
menu_center = None    # None = menu schowane
was_pinching = False  # czy w poprzedniej klatce było szczypnięcie
fire_particles = []   # cząsteczki ognia (pamiętane między klatkami)
active = None         # aktywna technika (None = żadna)
active_since = 0      # kiedy technika została włączona

while True:
    ok, frame = cap.read()
    if not ok:
        failures += 1
        if failures > 30:
            print("Cannot read frame")
            break
        continue
    failures = 0

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    picture = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect_for_video(picture, int(time.time() * 1000))

    h, w, _ = frame.shape

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        points = [(int(p.x * w), int(p.y * h)) for p in hand]
        index = points[8]

        # środek dłoni = średnia z nadgarstka i nasad 4 palców
        palm_ids = [0, 5, 9, 13, 17]
        palm = (sum(points[i][0] for i in palm_ids) // 5,
                sum(points[i][1] for i in palm_ids) // 5)

        # dłoń właśnie się pojawiła -> menu otwiera się pod palcem
        if menu_center is None:
            menu_center = (min(max(index[0], MENU_RADIUS), w - MENU_RADIUS),
                           min(max(index[1], MENU_RADIUS), h - MENU_RADIUS))

        selected = get_selected(menu_center, index)

        # szczypnięcie z "histerezą": łatwo złapać, trudno przypadkiem puścić
        hand_size = math.dist(points[0], points[9])
        ratio = math.dist(points[4], index) / hand_size
        if was_pinching:
            pinching = ratio < PINCH_OFF
        else:
            pinching = ratio < PINCH_ON

        # reagujemy tylko na POCZĄTEK szczypnięcia
        if pinching and not was_pinching:
            if active is not None:
                fire_particles.clear()  # nowy ogień zaczyna od zera
                print("Koniec:", active)
                active = None          # drugie szczypnięcie = wyłącz
                menu_center = None     # menu otworzy się od nowa pod palcem
            elif selected is not None:
                active = TECHNIQUES[selected]
                active_since = time.time()
                print("Jutsu:", active)
        was_pinching = pinching

        if active is None:
            if menu_center is not None:
                draw_menu(frame, menu_center, selected)
        elif active == "Rasengan":
            draw_rasengan(frame, palm, hand_size * 1.5, time.time())
        elif active == "Chidori":
            draw_chidori(frame, palm, hand_size * 1.5)
        elif active == "Katon":
            draw_katon(frame, palm, hand_size, fire_particles)
        elif active == "Kage Bunshin":
            seg = segmenter.segment_for_video(picture, int(time.time() * 1000))
            mask = seg.confidence_masks[0].numpy_view()
            # na wszelki wypadek ten sam rozmiar
            mask = cv2.resize(mask, (w, h))
            draw_clones(frame, mask, time.time() - active_since)

        cv2.circle(frame, index, 10, (0, 255, 0), -1)  # kursor = palec
    else:
        # dłoń zniknęła -> wszystko się resetuje
        menu_center = None
        was_pinching = False
        active = None
        fire_particles.clear()

    # przy aktywacji: błysk + wstrząs, potem napis w kolorze techniki
    if active is not None:
        elapsed = time.time() - active_since
        flash_and_shake(frame, TECH_COLORS[active], elapsed)
        draw_title(frame, active, elapsed)

    # licznik FPS (wygładzony, żeby nie skakał)
    now = time.time()
    fps = 0.9 * fps + 0.1 / max(now - last_time, 1e-6)
    last_time = now
    draw_text_outline(frame, f"FPS: {fps:.0f}",
                      (10, h - 15), 0.6, (0, 255, 0), 1)

    cv2.imshow("Ninjutsu", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
detector.close()
segmenter.close()
cv2.destroyAllWindows()
