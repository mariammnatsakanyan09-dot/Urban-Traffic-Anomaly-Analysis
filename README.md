![CI](https://github.com/mariammnatsakanyan09-dot/Urban-Traffic-Anomaly-Analysis/actions/workflows/ci.yml/badge.svg)

# Urban Traffic & Anomaly Analysis

An end-to-end MVP for a smart-city traffic monitoring service: a FastAPI backend that detects vehicles and pedestrians and tracks vehicles in dashcam footage, monitors incoming data for drift (night/rain conditions), and ships as a Docker container.

## Features

- **Object detection** — YOLOv8s in both the live API (`app.py`) and the offline batch scripts (`run_detection.py`, `run_tracking.py`), chosen for better recall than YOLOv8n at the cost of some speed. Cars, trucks, buses, motorcycles, and pedestrians are detected frame-by-frame with bounding boxes.
- **Multi-object tracking** (ByteTrack, built into Ultralytics) — assigns a persistent ID to each object and counts line crossings to estimate traffic flow and direction. Tracking covers cars, motorcycles, buses, and trucks; pedestrians are detected by the API but not tracked. `run_tracking.py` additionally detects low-light footage automatically and enhances those frames (CLAHE + gamma correction) before detection, improving recall at night without needing a separate model.
- **REST + WebSocket API** (FastAPI) — `POST /detect_frame` accepts either a single image or a short video (for a video, only the first frame is analyzed — see Testing the endpoints below); `WebSocket /detect_stream` streams a full video in real time with live stats.
- **Data validation** (Pydantic) — typed response schemas for `/detect_frame` (visible in `/docs`) and strict validation of the `/detect_stream` request: only plain file names with a known video extension are accepted, so a client cannot read files outside `data/samples/`.
- **Data drift monitoring** — `/detect_frame` compares each image against one fixed, pre-calibrated baseline; `/detect_stream` instead auto-calibrates a fresh baseline from each connection's own first ~30 frames, so it adapts to that camera's own viewing angle instead of being judged against an unrelated reference. Alerts are logged to `drift_log.jsonl`.
- **Inference optimization** — the model is exported to ONNX and benchmarked against the native PyTorch model (see Benchmarks below).
- **Containerized** — `Dockerfile` + `docker-compose.yml` for one-command deployment.
- **CI** — GitHub Actions builds the Docker image and runs an import smoke test on every push.
- **Data versioning** — large raw video files are tracked with DVC against a Google Drive remote (see "Getting video data" below).

## Architecture

```
dashcam video / uploaded image
        |
        v
 FastAPI backend (app.py)
   |- detector.py       -> YOLOv8s inference + bounding-box drawing
   |- drift_monitor.py  -> brightness/contrast drift detection
                            (fixed baseline for /detect_frame,
                             auto-calibrated per connection for /detect_stream)
        |
        v
 JSON response / WebSocket stream
   (detections, traffic stats, drift status, annotated frame)
```

Tracking (`tracker.py`, `run_tracking.py`) runs as a standalone batch script rather than inside the API, since line-crossing counts are computed over a full video rather than per single request.

## How to run

### Option A — Docker (recommended)

```bash
docker compose up --build
```

Then open http://localhost:8000/docs for the interactive API docs (Swagger UI).

### Option B — Local Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

### Getting video data

Two short sample videos (one daytime, one nighttime) are committed directly under `data/samples/`. The batch scripts (`run_detection.py`, `run_tracking.py`) and the live API's `/detect_stream` endpoint all read from `data/samples/`, so everything works immediately after cloning with no extra setup.

The full raw dataset (8 videos) is version-controlled with [DVC](https://dvc.org) against a personal Google Drive remote. Because that remote uses a personal (non-Workspace) OAuth client in testing mode, `dvc pull` only works for Google accounts explicitly added as test users — available on request. After `dvc pull` the files land in `data/raw/`; copy any video you want to stream through the API into `data/samples/`.

### Testing the endpoints

- **`POST /detect_frame`** — open `/docs`, click "Try it out" on this endpoint, and upload either an image or a short video. If a video is uploaded, only its first frame is analyzed (the response includes a `note` field saying so); use `/detect_stream` for full video processing.
- **`WebSocket /detect_stream`** — open `test_stream.html` directly in a browser, enter a video filename available under `data/samples` (the default, `13708052_2160_3840_30fps.mp4`, is one of the bundled samples), click "Start".

## Scripts

| Script | Purpose |
|---|---|
| `run_detection.py` | Batch-run detection (YOLOv8s) over videos in `data/samples` by default; saves annotated output and prints an FPS benchmark |
| `run_tracking.py` | Batch-run ByteTrack tracking (YOLOv8s) over videos in `data/samples`; auto-enhances low-light frames before detection and counts line crossings |
| `calibrate_baseline.py` | Computes the brightness/contrast baseline used by `/detect_frame`'s drift monitoring |
| `extract_frames.py` | Pulls single test frames out of videos (used for manual API testing) |
| `benchmark_onnx.py` | Compares PyTorch vs. ONNX Runtime inference speed |

## Benchmarks

Measured on an Apple M1 Pro (CPU only), 150 frames, downscaled to 1280px max width:

| Backend | FPS |
|---|---|
| YOLOv8n — PyTorch (.pt) | 19.59 |
| YOLOv8n — ONNX Runtime | 20.42 |

The gain from ONNX is modest here because PyTorch's CPU kernels are already well optimized for Apple Silicon; a larger improvement would be expected on a typical x86 CPU without equivalent native acceleration. A YOLOv8s comparison was also run with `benchmark_onnx.py` to quantify the nano-vs-small speed/accuracy trade-off used elsewhere in the project.

## Known limitations & future work

- **Dashcam footage, not a static camera.** The sample videos are filmed from a moving vehicle, so the line-crossing counts demonstrate the tracking algorithm rather than represent real traffic statistics from a fixed street camera — the same code would produce meaningful counts on genuine static traffic-camera footage.
- **Model size.** The API and the batch scripts all use YOLOv8s for better recall. YOLOv8n is a faster, less accurate alternative (see Benchmarks) if throughput on weaker hardware matters more than recall.
- **CI only, no CD.** GitHub Actions builds and smoke-tests the Docker image on every push, but there is no cloud deployment target configured yet to auto-deploy to.
- **Personal Google Drive DVC remote** doesn't scale to arbitrary reviewers without being added as an OAuth test user; a production setup would use a shared drive or a cloud bucket (S3/GCS) instead.
- **Synchronous inference inside the WebSocket loop** is fine for a single demo client, but would need `run_in_executor` (or a dedicated worker process) to serve multiple concurrent streams without blocking the event loop.

## Datasets referenced

Sample dashcam footage is used here for demonstration. For training or fine-tuning a custom detector at scale, these public datasets were used as reference/inspiration for the project's design:

- [BDD100K](https://bdd-data.berkeley.edu/) — Berkeley DeepDrive, a large-scale and diverse driving video dataset.
- [MIO-TCD](https://tcd.miovision.com/) — a traffic-camera dataset for vehicle classification and localization.
- [Cityscapes](https://www.cityscapes-dataset.com/) — an urban street-scene segmentation dataset.

## Tech stack

Python, FastAPI, Ultralytics YOLOv8, ByteTrack, OpenCV, ONNX Runtime, DVC, Docker, GitHub Actions.