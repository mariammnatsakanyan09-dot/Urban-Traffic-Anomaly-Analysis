import glob
import time
import cv2
from detector import load_model, detect_frame

VIDEO_PATH = sorted(glob.glob("data/raw/*.mp4"))[0]
MAX_FRAMES = 150
MAX_WIDTH = 1280


def benchmark(weights_path, video_path, max_frames=MAX_FRAMES):
    model = load_model(weights_path)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    start = time.time()
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        scale = min(1.0, MAX_WIDTH / w)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        detect_frame(model, frame)
        frame_count += 1
    elapsed = time.time() - start
    cap.release()
    fps = frame_count / elapsed if elapsed > 0 else 0
    return frame_count, elapsed, fps


def main():
    print(f"Видео для теста: {VIDEO_PATH}\n")

    print("PyTorch (.pt):")
    fc, el, fps = benchmark("yolov8n.pt", VIDEO_PATH)
    print(f"  {fc} кадров, {el:.1f} сек, {fps:.2f} FPS\n")

    print("ONNX (onnxruntime):")
    fc, el, fps = benchmark("yolov8n.onnx", VIDEO_PATH)
    print(f"  {fc} кадров, {el:.1f} сек, {fps:.2f} FPS\n")


if __name__ == "__main__":
    main()