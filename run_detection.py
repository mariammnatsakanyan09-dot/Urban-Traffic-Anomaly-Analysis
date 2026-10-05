import glob
import os
import time
import cv2
from detector import load_model, detect_frame, draw_detections

INPUT_DIR = "data/samples"
OUTPUT_DIR = "data/processed"
VIDEO_PATTERNS = ("*.mp4", "*.mov", "*.avi", "*.webm")
MAX_VIDEOS = 3          
MAX_WIDTH = 1280     


def process_video(model, video_path, output_path, max_width=MAX_WIDTH):
    cap = cv2.VideoCapture(video_path)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    scale = min(1.0, max_width / orig_width)
    width, height = int(orig_width * scale), int(orig_height * scale)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps_in, (width, height))

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if scale < 1.0:
            frame = cv2.resize(frame, (width, height))
        detections = detect_frame(model, frame)
        frame = draw_detections(frame, detections)
        writer.write(frame)
        frame_count += 1

        if frame_count % 10 == 0 or frame_count == total_frames:
            elapsed_so_far = time.time() - start_time
            fps_so_far = frame_count / elapsed_so_far if elapsed_so_far > 0 else 0
            print(f"\r  {frame_count}/{total_frames} кадров, {fps_so_far:.2f} FPS", end="", flush=True)

    print()
    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed if elapsed > 0 else 0

    cap.release()
    writer.release()
    return frame_count, elapsed, avg_fps


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = load_model("yolov8s.pt")

    video_files = []
    for pattern in VIDEO_PATTERNS:
        video_files.extend(glob.glob(os.path.join(INPUT_DIR, pattern)))

    if not video_files:
        print("Видео не найдены в data/raw — проверь путь и расширения файлов.")
        return

    video_files = sorted(video_files)[:MAX_VIDEOS]
    print(f"Обрабатываю {len(video_files)} из найденных видео (ограничение MAX_VIDEOS={MAX_VIDEOS})")

    for video_path in video_files:
        name = os.path.basename(video_path)
        output_path = os.path.join(OUTPUT_DIR, f"annotated_{name}")
        print(f"\n{name}:")
        frame_count, elapsed, avg_fps = process_video(model, video_path, output_path)
        print(f"  Готово: {frame_count} кадров, {elapsed:.1f} сек, {avg_fps:.2f} FPS")


if __name__ == "__main__":
    main()