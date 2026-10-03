import base64
import asyncio
import json

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from detector import load_model, detect_frame, draw_detections, TARGET_CLASSES

app = FastAPI(title="Urban Traffic Monitor API")

model = load_model("yolov8n.onnx")  # модель грузим один раз при старте сервера
from drift_monitor import DriftMonitor

drift_monitor = DriftMonitor(baseline_brightness=118.3, baseline_contrast=52.1)
# ^ подставь СВОИ числа с шага калибровки

def compute_stats(detections):
    counts = {}
    for (_, _, _, _, cls_id, _) in detections:
        name = TARGET_CLASSES[cls_id]
        counts[name] = counts.get(name, 0) + 1

    total_vehicles = sum(v for k, v in counts.items() if k != "person")

    # простая эвристика загруженности — для MVP этого достаточно
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


@app.post("/detect_frame")
async def detect_frame_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse(status_code=400, content={"error": "Не удалось прочитать изображение"})

    detections = detect_frame(model, frame)
    drift_result = drift_monitor.check(frame)
    annotated = draw_detections(frame.copy(), detections)

    boxes_json = [
        {
            "class": TARGET_CLASSES[cls_id],
            "confidence": round(conf, 3),
            "bbox": [x1, y1, x2, y2],
        }
        for (x1, y1, x2, y2, cls_id, conf) in detections
    ]

    return {
        "detections": boxes_json,
        "stats": compute_stats(detections),
        "drift": drift_result,
        "annotated_image_base64": encode_frame_to_base64(annotated),
    }


@app.websocket("/detect_stream")
async def detect_stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    cap = None
    try:
        init_message = await websocket.receive_text()
        params = json.loads(init_message)
        video_name = params.get("video")
        video_path = f"data/raw/{video_name}"

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            await websocket.send_json({"error": f"Не удалось открыть видео: {video_path}"})
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                await websocket.send_json({"event": "end_of_stream"})
                break

            detections = detect_frame(model, frame)
            drift_result = drift_monitor.check(frame)
            annotated = draw_detections(frame.copy(), detections)

            await websocket.send_json({
                "frame": encode_frame_to_base64(annotated),
                "stats": compute_stats(detections),
                "drift": drift_result,
            })

            await asyncio.sleep(0.03)  # небольшой буфер, не душим event loop

    except WebSocketDisconnect:
        pass
    finally:
        if cap is not None:
            cap.release()