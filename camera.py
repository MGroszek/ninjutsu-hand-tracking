import cv2      # biblioteka OpenCV do obrazu i kamery

cap = cv2.VideoCapture(0)  # 0 = domyślna kamera w Macu

while True:
    ok, frame = cap.read()  # pobierz jedną klatkę
    if not ok:
        print("Cannot read frame")
        break

    frame = cv2.flip(frame, 1)    # odbicie lustrzane
    cv2.imshow("Ninjustu", frame)  # pokaż klatkę w oknie

    if cv2.waitKey(1) & 0xFF == ord('q'):  # klawisz q = koniec
        break

cap.release()   # zwolnij kamerę
cv2.destroyAllWindows()  # zamknij okno
