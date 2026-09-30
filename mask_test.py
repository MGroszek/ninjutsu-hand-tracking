import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

option = vision.ImageSegmenterOptions(
    base_options=BaseOptions(
        model_asset_path="selfie_segmenter.tflite",
        delegate=BaseOptions.Delegate.CPU,
    ),
    running_mode=vision.RunningMode.VIDEO,
    output_confidence_masks=True,
)
segmenter = vision.ImageSegmenter.create_from_options(option)

cap = cv2.VideoCapture(0)
failures = 0
printed = False

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
    result = segmenter.segment_for_video(picture, int(time.time() * 1000))

    masks = result.confidence_masks
    if not printed:  # raz wypisz, co dostaliśmy
        print("Liczba masek:", len(masks))
        printed = True

    # maska: liczby 0..1 dla każdego piksela (pewność, że to człowiek)
    mask = masks[0].numpy_view()
    cv2.imshow("Maska", (mask * 255).astype(np.uint8))
    cv2.imshow("Kamera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
segmenter.close()
cv2.destroyAllWindows()
