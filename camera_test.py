import cv2
import time

for numer in range(3):
    print(f"---Kamera {numer}---")
    # 0 = domyślna kamera w Macu
    cap = cv2.VideoCapture(numer, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        print(f"Nie można otworzyć kamery {numer}")
        continue

    for proba in range(20):
        ok, frame = cap.read()  # pobierz jedną klatkę
        if ok:
            print(f"Nie można odczytać klatki z kamery {numer}")
            break
        time.sleep(0.1)  # poczekaj 100 ms

    else:
        print("Otwarta ale nie wysyla obrazów")

    cap.release()   # zwolnij kamerę
