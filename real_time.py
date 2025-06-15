"""
Скрипт для тестирования алгоритма: запись видео в файл, инференс через YOLO с заданным интервалом,
сохранение кадров с детекциями и лог в CSV с геопозицией через CoreLocationCLI.

Функционал:
1. Захват потока с камеры и запись в видеофайл (OpenCV VideoWriter).
2. Обработка кадров через модель YOLO раз в `INTERVAL` секунд.
3. При наличии предсказаний:
   - сохранение аннотированного кадр в папку `outputs/` под именем `detect_{i}.jpg`;
   - запись в CSV `reports/detections.csv` строки с timestamp, class_name, filename, lat, lon.
"""

import cv2
import time
import csv
import subprocess
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# ======================
# Функция для получения координат
# ======================
def get_location():
    """
    Вызывает CoreLocationCLI и возвращает (lat, lon) или (None, None).
    CoreLocationCLI формат: "%latitude %longitude" -> "lat lon".
    """
    try:
        cmd = ["/opt/homebrew/bin/CoreLocationCLI", "-once", "-format", "%latitude %longitude"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=2.0)
        out = res.stdout.strip()
        parts = out.split()
        if len(parts) >= 2:
            return float(parts[0]), float(parts[1])
    except Exception:
        pass
    return None, None

# ======================
# Параметры и инициализация
# ======================
CAMERA_INDEX = 0
INTERVAL     = 1.0  # секунды между инференсом
OUTPUT_DIR   = Path('outputs')
REPORT_DIR   = Path('reports')
VIDEO_FILE   = REPORT_DIR / 'session.avi'
CSV_FILE     = REPORT_DIR / 'detections.csv'

# Подготовка директорий
OUTPUT_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# Инициализация CSV
csv_file   = open(CSV_FILE, 'w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
# Заголовок с геопозицией
csv_writer.writerow(['timestamp', 'class', 'filename', 'confidence', 'lat', 'lon'])

# Инициализация видео-записи
cap = cv2.VideoCapture(CAMERA_INDEX)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out    = cv2.VideoWriter(str(VIDEO_FILE), fourcc, fps, (width, height))

# Загрузка модели
model = YOLO('best_my.pt')  

last_time    = time.time()
detect_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Записываем необработанный кадр в видео
        out.write(frame)

        now = time.time()
        if now - last_time >= INTERVAL:
            last_time = now

            results = model.predict(frame, conf=0.4)
            res     = results[0]
            if len(res.boxes) > 0:
                # Отрисовка боксов
                annotated = res.plot()
                detect_count += 1
                filename = f'detect_{detect_count}.jpg'
                filepath = OUTPUT_DIR / filename
                cv2.imwrite(str(filepath), annotated)

                # Получаем текущие координаты
                lat, lon = get_location()
                timestamp = datetime.now().isoformat()

                # Логируем каждую детекцию вместе с confidence
                for cls, conf in zip(res.boxes.cls, res.boxes.conf):
                    class_name = model.names[int(cls)]
                    timestamp  = datetime.now().isoformat()
                    csv_writer.writerow([
                        timestamp,
                        class_name,
                        filename,
                        f"{float(conf):.3f}",
                        f"{lat:.6f}" if lat is not None else "",
                        f"{lon:.6f}" if lon is not None else ""
                    ])
                csv_file.flush()

        # Показать необработанное видео
        cv2.imshow('Real-Time Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    csv_file.close()
