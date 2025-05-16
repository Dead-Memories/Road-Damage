import cv2

from ultralytics import YOLO

import cv2
import time
from ultralytics import YOLO

model = YOLO("best.pt")  # или твоя модель

cap = cv2.VideoCapture(0)

last_processed = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    if now - last_processed >= 1.0:  # раз в секунду
        last_processed = now

        results = model.predict(frame, conf=0.4)
        annotated = results[0].plot()

        # Можно сохранить кадр или вывести классы:
        # results[0].boxes.xyxy, results[0].boxes.cls

        cv2.imshow("YOLOv8 Real-Time (1 FPS)", annotated)
    else:
        # Просто показываем обычный фрейм (без YOLO)
        cv2.imshow("YOLOv8 Real-Time (1 FPS)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
