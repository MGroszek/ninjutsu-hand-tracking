# Ninjutsu – Hand-Tracking Jutsu Effects 🍥

Control Naruto-style techniques with your hand in front of a webcam.
A radial menu follows your hand, you point at a technique and **pinch** (thumb + index finger) to cast it.

Built with **Python**, **OpenCV** and **MediaPipe**. All visual effects are generated in real time with code (no video or image assets).

## Techniques

| Menu | Technique | Effect |
|---|---|---|
| ⬆️ Up | **Rasengan** | Shaded "3D" energy sphere with a rotating surface texture and energy trails orbiting in 3D |
| ➡️ Right | **Chidori** | Branching 3D lightning, closer bolts are thicker and brighter (depth sorting) |
| ⬇️ Down | **Katon** | Particle fire with depth, a hot core and rising smoke |
| ⬅️ Left | **Kage Bunshin** | Two shadow clones behind you (person segmentation) with a smoke puff |

Every cast also triggers a colour flash, screen shake and an animated title.

## How it works

1. **Hand tracking** – MediaPipe Hand Landmarker finds 21 points on the hand.
2. **Radial menu** – the angle between the menu centre and the index fingertip selects a slice.
3. **Pinch detection** – thumb–index distance divided by hand size (works at any distance), with **hysteresis** (two thresholds) so it does not flicker.
4. **Effects** – drawn with OpenCV: glow via blurred layers and additive blending, fake 3D via per-pixel shading and depth (z) for size, brightness and draw order.
5. **Kage Bunshin** – MediaPipe Selfie Segmenter separates the person from the background; the person is copied, scaled and blended behind the original.

## Setup (macOS, Apple Silicon)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the two MediaPipe models into the project folder:

```bash
curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
curl -L -o selfie_segmenter.tflite https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite
```

Run:

```bash
.venv/bin/python menu.py
```

On the first run macOS asks for camera permission for your terminal – allow it and start again.

## Controls

- Show your hand → the menu opens under your index finger
- Move the finger to a slice → it lights up
- Pinch → cast the technique, pinch again → stop
- `q` → quit (click the video window first)

## Notes

- **MediaPipe 1.0.1 crashes on macOS** (Metal / `TensorsToDetectionsCalculator`, [issue #6356](https://github.com/google-ai-edge/mediapipe/issues/6356)). This project uses **mediapipe 1.0.0**.
- Tested on MacBook Pro M4 Pro, Python 3.14, ~30 FPS at 1280×720.

## Other files

- `camera.py`, `camera_test.py` – camera tests
- `hands.py` – hand landmarks + pinch test
- `mask_test.py` – segmentation mask test


## Roadmap

- Close a technique with a fist
- Version 2: Python sends hand data via OSC to **Unreal Engine 5**, effects rendered with Niagara