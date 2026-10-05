import base64
import asyncio
import json
import os
import tempfile

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from detector import load_model, detect_frame, draw_detections, TARGET_CLASSES
from drift_monitor import DriftMonitor

app = FastAPI(title="Urban Traffic Monitor API")

model = load_model("yolov8s.pt")  # load the model once at server startup

# Fixed baseline for single-image checks via /detect_frame, calibrated from calibrate_baseline.py.
# For continuous streams (/detect_stream), each connection gets its own auto-calibrating
# DriftMonitor instead — see detect_stream_endpoint below — so it adapts to that camera's
# own viewing angle instead of being compared against this one fixed reference.
drift_monitor = DriftMonitor(baseline_brightness=116.9, baseline_contrast=49.8)

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".webm", ".mkv")


def compute_stats(detections):
    counts = {}
    for (_, _, _, _, cls_id, _) in detections:
        name = TARGET_CLASSES[cls_id]
        counts[name] = counts.get(name, 0) + 1

    total_vehicles = sum(v for k, v in counts.items() if k != "person")

    # simple congestion heuristic — good enough for an MVP
    if total_vehicles >= 8:
        load = "high"
    elif total_vehicles >= 4:
        load = "medium"
    else:
        load = "low"

    return {"counts": counts, "total_vehicles": total_vehicles, "load": load}


def encode_frame_to_base64(frame):
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return None
    return base64.b64encode(buffer).decode("utf-8")


def read_first_frame_from_video_bytes(contents, suffix):
    """Writes the uploaded video to a temp file and reads just its first frame.
    cv2.VideoCapture needs a real file path, it cannot read from an in-memory buffer."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    try:
        cap = cv2.VideoCapture(tmp_path)
        ret, frame = cap.read()
        cap.release()
    finally:
        os.remove(tmp_path)
    return frame if ret else None


@app.post("/detect_frame")
async def detect_frame_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    filename = (file.filename or "").lower()
    note = None

    if filename.endswith(VIDEO_EXTENSIONS):
        suffix = os.path.splitext(filename)[1]
        frame = read_first_frame_from_video_bytes(contents, suffix)
        note = "A video was uploaded — only its first frame was analyzed. Use /detect_stream for full video processing."
    else:
        np_arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse(status_code=400, content={"error": "Could not read an image or a video frame from the uploaded file"})

    detections = detect_frame(model, frame)
    annotated = draw_detections(frame.copy(), detections)
    drift_result = drift_monitor.check(frame)

    boxes_json = [
        {
            "class": TARGET_CLASSES[cls_id],
            "confidence": round(conf, 3),
            "bbox": [x1, y1, x2, y2],
        }
        for (x1, y1, x2, y2, cls_id, conf) in detections
    ]

    response = {
        "detections": boxes_json,
        "stats": compute_stats(detections),
        "drift": drift_result,
        "annotated_image_base64": encode_frame_to_base64(annotated),
    }
    if note:
        response["note"] = note
    return response


@app.websocket("/detect_stream")
async def detect_stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    cap = None
    # Each stream gets its own monitor that auto-calibrates from THIS video's first frames,
    # instead of sharing one fixed baseline — this is what makes drift detection work
    # regardless of camera angle (aerial, dashcam, fixed street camera, etc).
    stream_drift_monitor = DriftMonitor()
    try:
        init_message = await websocket.receive_text()
        params = json.loads(init_message)
        video_name = params.get("video")
        video_path = f"data/raw/{video_name}"

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            await websocket.send_json({"error": f"Could not open video: {video_path}"})
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                await websocket.send_json({"event": "end_of_stream"})
                break

            detections = detect_frame(model, frame)
            drift_result = stream_drift_monitor.check(frame)
            annotated = draw_detections(frame.copy(), detections)

            await websocket.send_json({
                "frame": encode_frame_to_base64(annotated),
                "stats": compute_stats(detections),
                "drift": drift_result,
            })

            await asyncio.sleep(0.03)  # small buffer so we don't choke the event loop

    except WebSocketDisconnect:
        pass
    finally:
        if cap is not None:
            cap.release()