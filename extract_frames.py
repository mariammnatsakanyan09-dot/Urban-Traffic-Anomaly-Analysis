import cv2

def extract_frame(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.imwrite(output_path, frame)
        print(f"Сохранено: {output_path}")
    else:
        print(f"Не удалось прочитать кадр из {video_path}")

extract_frame("data/raw/15446272_2160_3840_60fps.mp4", "frame_day.jpg")
extract_frame("data/raw/16520752_2560_1440_60fps.mp4", "frame_night.jpg")