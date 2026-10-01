import math
import socket
import time
from pythonosc.udp_client import SimpleUDPClient


def local_ip():
    """Sprawdza aktualny adres tego Maca w sieci."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))   # nic nie wysyła, tylko pyta system o trasę
    ip = s.getsockname()[0]
    s.close()
    return ip


ip = local_ip()
print("Wysyłam do:", ip)
client = SimpleUDPClient(ip, 8000)
t = 0.0
while True:
    x = 0.5 + 0.4 * math.sin(t)   # liczba płynnie od 0.1 do 0.9 i z powrotem
    y = 0.5
    client.send_message("/hand", [x, y])
    print(f"wysłano /hand {x:.2f} {y:.2f}")
    t += 0.1
    time.sleep(0.1)               # 10 wiadomości na sekundę
