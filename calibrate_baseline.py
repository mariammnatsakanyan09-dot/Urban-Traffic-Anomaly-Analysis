import glob
import cv2
import numpy as np

SAMPLE_FRAMES = 30  # по сколько кадров из каждого видео брать для оценки


def video_stats(video_path, sample_frames=SAMPLE_FRAMES):
    cap = cv2.VideoCapture(video_path)
    brightness_values, contrast_values = [], []
    count = 0
    while count < sample_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_values.append(np.mean(gray))
        contrast_values.append(np.std(gray))
        count += 1
    cap.release()
    if not brightness_values:
        return None
    return np.mean(brightness_values), np.mean(contrast_values)


def main():
    videos = sorted(glob.glob("data/raw/*.mp4"))
    for video_path in videos:
        stats = video_stats(video_path)
        if stats:
            brightness, contrast = stats
            print(f"{video_path}: brightness={brightness:.1f}, contrast={contrast:.1f}")


if __name__ == "__main__":
    main()