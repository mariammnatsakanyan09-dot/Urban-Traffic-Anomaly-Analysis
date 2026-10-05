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

extract_frame("data/samples/13708052_2160_3840_30fps.mp4", "frame_day.jpg")
extract_frame("data/samples/19811275-hd_1920_1080_30fps.mp4", "frame_night.jpg")