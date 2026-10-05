import glob
import os
import time
import cv2
import numpy as np
from tracker import load_model, LineCounter, TARGET_CLASSES, COLORS


INPUT_DIR = "data/samples"
OUTPUT_DIR = "data/tracked"
VIDEO_PATTERNS = ("*.mp4", "*.mov", "*.avi", "*.webm")
MAX_VIDEOS = 2
MAX_WIDTH = 1600
CONF_THRESHOLD = 0.10          
IMGSZ = 1280


def is_dark_video(video_path, sample_frames=12):
    cap = cv2.VideoCapture(video_path)
    brightness = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    step = max(1, total // sample_frames)

    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness.append(np.mean(gray))

    cap.release()
    avg = np.mean(brightness) if brightness else 128
    return avg < 75


def enhance_night_frame(frame):
    """Сильное усиление для ночных кадров"""
    # 1. CLAHE
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # 2. Более агрессивная гамма
    gamma = 1.4
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    enhanced = cv2.LUT(enhanced, table)

    # 3. Небольшое увеличение контраста
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.15, beta=10)

    return enhanced


def process_video(model, video_path, output_path, max_width=MAX_WIDTH):
    cap = cv2.VideoCapture(video_path)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    scale = min(1.0, max_width / orig_width)
    width = int(orig_width * scale)
    height = int(orig_height * scale)

    dark = is_dark_video(video_path)
    if dark:
        print("  → Тёмное видео: усиление применяется ПЕРЕД детекцией")

    counter = LineCounter(line_y_ratio=0.55)
    counter.set_frame_size(height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps_in, (width, height))

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

       
        if dark:
            frame_for_detect = enhance_night_frame(frame)
        else:
            frame_for_detect = frame

        if scale < 1.0:
            frame_for_detect = cv2.resize(frame_for_detect, (width, height))
            frame = cv2.resize(frame, (width, height))   

        # Детекция + трекинг на одном кадре
        results = model.track(
            source=frame_for_detect,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
            conf=CONF_THRESHOLD,
            imgsz=IMGSZ,
            classes=[2, 3, 5, 7],   # car, motorcycle, bus, truck
        )

        result = results[0]

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int)
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls_id in zip(boxes, ids, classes):
                if cls_id not in TARGET_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                direction = counter.update(track_id, cx, cy)

                color = COLORS.get(cls_id, (255, 255, 255))
                label = f"{TARGET_CLASSES[cls_id]} #{track_id}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 4, color, -1)
                cv2.putText(frame, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                if direction:
                    print(f"  Пересечение: ID {track_id} ({TARGET_CLASSES[cls_id]}) -> {direction}")

        counter.draw_line_and_counts(frame, width)
        writer.write(frame)
        frame_count += 1

        if frame_count % 15 == 0 or frame_count == total_frames:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"\r  {frame_count}/{total_frames} кадров, {fps:.2f} FPS", end="", flush=True)

    print()
    cap.release()
    writer.release()
    elapsed = time.time() - start_time
    return frame_count, elapsed, counter


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = load_model("yolov8s.pt")   

    video_files = []
    for pattern in VIDEO_PATTERNS:
        video_files.extend(glob.glob(os.path.join(INPUT_DIR, pattern)))

    if not video_files:
        print("Видео не найдены в data/samples.")
        return

    video_files = sorted(video_files)[:MAX_VIDEOS]
    print(f"Обрабатываю {len(video_files)} видео с трекингом")

    for video_path in video_files:
        name = os.path.basename(video_path)
        output_path = os.path.join(OUTPUT_DIR, f"tracked_{name}")
        print(f"\n{name}:")
        frame_count, elapsed, counter = process_video(model, video_path, output_path)
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"  Готово: {frame_count} кадров, {elapsed:.1f} сек, {avg_fps:.2f} FPS")
        print(f"  Итог: down={counter.count_down}, up={counter.count_up}, total={counter.count_down + counter.count_up}")


if __name__ == "__main__":
    main()