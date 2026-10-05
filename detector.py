import cv2
from ultralytics import YOLO


TARGET_CLASSES = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

COLORS = {
    0: (255, 200, 0),
    2: (0, 255, 0),
    3: (0, 200, 255),
    5: (255, 0, 0),
    7: (0, 0, 255),
}


def load_model(weights_path="yolov8s.pt"):
    return YOLO(weights_path)


def detect_frame(model, frame, conf_threshold=0.35):
    """Возвращает список (x1, y1, x2, y2, class_id, confidence) только по нужным классам."""
    results = model(frame, verbose=False)[0]
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id in TARGET_CLASSES and conf >= conf_threshold:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append((x1, y1, x2, y2, cls_id, conf))
    return detections


def draw_detections(frame, detections):
    for (x1, y1, x2, y2, cls_id, conf) in detections:
        color = COLORS.get(cls_id, (255, 255, 255))
        label = f"{TARGET_CLASSES[cls_id]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return frame