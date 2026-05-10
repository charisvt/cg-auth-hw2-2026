# CG HW2 — 3D Transformations & Projections

Implements a 3D-to-2D rendering pipeline: affine transformations, camera lookat, perspective projection, rasterization, and Gouraud shading.

## Files

| File                 |
|----------------------|
| `transformations.py` |
| `renderer.py`        |
| `demo1.py`           |
| `demo2.py`           |

## Requirements

Python >= 3.12, `numpy`, `Pillow`

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy pillow
```

## Run

```bash
python demo1.py   # outputs frames_demo1/
python demo2.py   # outputs frames_demo2/
```

## Create videos

```bash
ffmpeg -r 25 -i frames_demo1/frame_%03d.png -vcodec libx264 demo1.mp4
ffmpeg -r 25 -i frames_demo2/frame_%03d.png -vcodec libx264 demo2.mp4
```
