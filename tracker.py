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


def load_model(weights_path="yolov8n.pt"):
    return YOLO(weights_path)


class LineCounter:
    """Считает пересечения горизонтальной линии треками объектов."""

    def __init__(self, line_y_ratio=0.6):
        self.line_y_ratio = line_y_ratio  # линия на 60% высоты кадра
        self.line_y = None
        self.prev_centers = {}   # track_id -> предыдущий центр (cx, cy)
        self.count_down = 0      # сверху вниз ("приближается" к камере)
        self.count_up = 0        # снизу вверх ("удаляется")

    def set_frame_size(self, height):
        self.line_y = int(height * self.line_y_ratio)

    def update(self, track_id, cx, cy):
        """Возвращает 'up'/'down'/None, если произошло пересечение линии."""
        direction = None
        prev = self.prev_centers.get(track_id)
        if prev is not None:
            prev_cy = prev[1]
            if prev_cy < self.line_y <= cy:
                direction = "down"
                self.count_down += 1
            elif prev_cy > self.line_y >= cy:
                direction = "up"
                self.count_up += 1
        self.prev_centers[track_id] = (cx, cy)
        return direction

    def draw_line_and_counts(self, frame, width):
        cv2.line(frame, (0, self.line_y), (width, self.line_y), (0, 255, 255), 2)
        text = f"Down: {self.count_down}  Up: {self.count_up}  Total: {self.count_down + self.count_up}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)