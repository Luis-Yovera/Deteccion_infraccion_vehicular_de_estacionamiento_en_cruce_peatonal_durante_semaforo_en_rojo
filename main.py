import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import *
import easyocr
from util import write_csv

# Definir la región de interés (Video prueba con calidad 4k (2160 p))
px1, py1 = 0, 2160
px2, py2 = 2540, 2160
px3, py3 = 3000, 1500
px4, py4 = 1380, 1360
px5, py5 = 0, 1600

region_points = {
    "region-01": [(px1, py1), (px2, py2), (px3, py3), (px4, py4), (px5, py5)]
}

# Rangos de color rojo en HSV
lower_red1 = (0, 100, 100)
upper_red1 = (10, 255, 255)
lower_red2 = (170, 100, 100)
upper_red2 = (180, 255, 255)

# Funciones
def load_yolo_model():
    try:
        model = YOLO('best_entrenamiento_placas_v4.pt')
        return model
    except ImportError:
        print("Please install ultralytics: pip install ultralytics")
        return None

def create_roi_mask(frame, region_points):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    polygon = np.array(region_points, dtype=np.int32)
    cv2.fillPoly(mask, [polygon], 255)
    return mask

def apply_roi_detection(frame, model, region_points, reader):
    roi_mask = create_roi_mask(frame, region_points)
    roi_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
    resultsRoi = model(roi_frame)
    filtered_detections = []

    for result in resultsRoi:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            mask_value = cv2.pointPolygonTest(np.array(region_points, dtype=np.int32), (int(center_x), int(center_y)), False)
            
            if mask_value >= 0:
                plate_img = frame[int(y1):int(y2), int(x1):int(x2)]
                try:
                    ocr_results = reader.readtext(plate_img)
                    plate_text = max(ocr_results, key=lambda x: x[2], default=(None, None, 0))[1] if ocr_results else "No text"
                except Exception as e:
                    plate_text = "OCR Error"

                filtered_detections.append({
                    'box': box,
                    'class': model.names[int(box.cls[0])],
                    'confidence': float(box.conf[0]),
                    'plate_text': plate_text
                })

    return filtered_detections, roi_frame

results = {}

def main(reader):
    frame_nmr = 1
    car_id = 1

    cap = cv2.VideoCapture(r"C:\Users\tangi\Downloads\PROYECTO\VIDEOS\prueba_4k.mp4")
    cap_sem = cv2.VideoCapture(r"C:\Users\tangi\Downloads\PROYECTO\VIDEOS\semaforo_4k.mp4")

    assert cap.isOpened(), "Error al leer el archivo de video de tráfico"
    assert cap_sem.isOpened(), "Error al leer el archivo de video del semáforo"

    # Se obtiene los FPS y dimensiones de ambos videos
    w, h, fps_traffic = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    fps_sem = int(cap_sem.get(cv2.CAP_PROP_FPS))

    # Se sincronizan usando el menor FPS
    fps_output = min(fps_traffic, fps_sem)

    # Inicializar el escritor de video
    video_writer = cv2.VideoWriter("traffic_detection_results.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps_output, (w, h))

    RED_THRESHOLD = 200
    model = load_yolo_model()
    if not model:
        return

    while cap.isOpened():
        frame_nmr += 1
        car_id += 1

        ret, frame = cap.read()
        ret_sem, frame_sem = cap_sem.read()

        if not ret or not ret_sem:
            print("Fin del video")
            break

        # Redimensionar el frame del semáforo
        dim = (213, 216)
        resized_sem = cv2.resize(frame_sem, dim, interpolation=cv2.INTER_AREA)

        # Detectar semáforo rojo
        hsv_frame = cv2.cvtColor(resized_sem, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        red_pixel_count = cv2.countNonZero(mask)

        # Procesar si el semáforo está en rojo
        if red_pixel_count > RED_THRESHOLD:
            roi_points = region_points["region-01"]
            cv2.polylines(frame, [np.array(roi_points)], isClosed=True, color=(0, 255, 0), thickness=2)

            if frame is not None:
                detections, roi_frame = apply_roi_detection(frame, model, roi_points, reader)

                for detection in detections:
                    x1, y1, x2, y2 = detection['box'].xyxy[0]
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    license_plate = f"{detection['plate_text']}"
                    license_confidence = f"{detection['confidence']:.2f}"
                    cv2.putText(frame, license_plate + ' | ' + license_confidence, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                    if license_plate != "No text":
                        results[frame_nmr] = {
                            'license': {
                                'box': [x1, y1, x2, y2],
                                'plate': license_plate,
                                'confidence': license_confidence
                            }
                        }

        cv2.imshow('Detections', frame)
        video_writer.write(frame)
        cv2.imshow('Semaforo', resized_sem)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    write_csv(results, './traffic_detection_results.csv')
    cap.release()
    cap_sem.release()
    video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    reader = easyocr.Reader(['es'])
    main(reader)
