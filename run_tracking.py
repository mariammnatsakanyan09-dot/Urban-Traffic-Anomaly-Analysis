import glob
import os
import time
import cv2
from tracker import load_model, LineCounter, TARGET_CLASSES, COLORS

INPUT_DIR = "data/raw"
OUTPUT_DIR = "data/tracked"
VIDEO_PATTERNS = ("*.mp4", "*.mov", "*.avi", "*.webm")
MAX_VIDEOS = 2
MAX_WIDTH = 1280
CONF_THRESHOLD = 0.25


def process_video(model, video_path, output_path, max_width=MAX_WIDTH):
    cap = cv2.VideoCapture(video_path)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.release()  # дальше видео откроет сам model.track

    scale = min(1.0, max_width / orig_width)
    width, height = int(orig_width * scale), int(orig_height * scale)

    counter = LineCounter(line_y_ratio=0.6)
    counter.set_frame_size(height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps_in, (width, height))

    frame_count = 0
    start_time = time.time()

    results = model.track(
        source=video_path,
        tracker="bytetrack.yaml",
        persist=True,
        stream=True,
        verbose=False,
        conf=CONF_THRESHOLD,
    )

    for result in results:
        frame = result.orig_img
        if scale < 1.0:
            frame = cv2.resize(frame, (width, height))

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int)
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls_id in zip(boxes, ids, classes):
                if cls_id not in TARGET_CLASSES:
                    continue
                x1, y1, x2, y2 = box
                if scale < 1.0:
                    x1, y1, x2, y2 = x1 * scale, y1 * scale, x2 * scale, y2 * scale
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
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

        if frame_count % 10 == 0 or frame_count == total_frames:
            elapsed_so_far = time.time() - start_time
            fps_so_far = frame_count / elapsed_so_far if elapsed_so_far > 0 else 0
            print(f"\r  {frame_count}/{total_frames} кадров, {fps_so_far:.2f} FPS", end="", flush=True)

    print()
    writer.release()
    elapsed = time.time() - start_time
    return frame_count, elapsed, counter


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = load_model("yolov8n.pt")

    video_files = []
    for pattern in VIDEO_PATTERNS:
        video_files.extend(glob.glob(os.path.join(INPUT_DIR, pattern)))

    if not video_files:
        print("Видео не найдены в data/raw.")
        return

    video_files = sorted(video_files)[:MAX_VIDEOS]
    print(f"Обрабатываю {len(video_files)} видео с трекингом (MAX_VIDEOS={MAX_VIDEOS})")

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