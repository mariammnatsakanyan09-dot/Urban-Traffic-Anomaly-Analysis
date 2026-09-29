import torch
from ultralytics import YOLO

# 1. Проверяем доступность GPU (ускорителя видеокарты)
print("Доступен ли CUDA (GPU):", torch.cuda.is_available())

# 2. Загружаем предобученную модель YOLOv8n
model = YOLO('yolov8n.pt')

# 3. Выводим словарь со всеми классами датасета COCO
print("\nСписок классов модели:")
print(model.names)
 