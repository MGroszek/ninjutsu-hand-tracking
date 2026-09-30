import time
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
import math

option = vision.HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task",
        delegate=BaseOptions.Delegate.CPU,
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
)
detector = vision.HandLandmarker.create_from_options(option)
connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

cap = cv2.VideoCapture(0)
failures = 0

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

    time_ms = int(round(time.time() * 1000))
    result = detector.detect_for_video(picture, time_ms)

    h, w, _ = frame.shape
    for hand in result.hand_landmarks:
        points = [(int(point.x * w), int(point.y * h)) for point in hand]
        for connection in connections:
            cv2.line(frame, points[connection.start],
                     points[connection.end], (255, 255, 255), 2)
        for x, y in points:
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

            # --- szczypnięcie: kciuk (4) + palec wskazujący (8) ---
        thumb = points[4]
        index = points[8]
        hand_size = math.dist(points[0], points[9])
        pinch = math.dist(thumb, index) / hand_size

        cv2.putText(frame, f"{pinch:.2f}", (index[0] + 15, index[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        if pinch < 0.25:
            cv2.circle(frame, index, 20, (0, 0, 255), 3)
            cv2.putText(frame, "PINCH!", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.putText(frame, f"Hand count: {len(result.hand_landmarks)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Ninjustu", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
detector.close()
cv2.destroyAllWindows()
