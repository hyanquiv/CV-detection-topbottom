import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans

# --- Función para obtener el color dominante ---
def get_dominant_color(img, k=1):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(img)
    color = kmeans.cluster_centers_[0].astype(int)
    return tuple(color)

# --- Cargar modelo entrenado ---
model = YOLO("mi_yolov8_modelo.pt")  # tu modelo .pt entrenado

# --- Abrir video o webcam ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        class_id = int(cls)

        label = ["top", "bottom", "exposed"][class_id]
        roi = frame[y1:y2, x1:x2]

        if label != "exposed" and roi.size > 0:
            color = get_dominant_color(roi)
            color_text = f"RGB({color[0]},{color[1]},{color[2]})"
        else:
            color_text = ""

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
        cv2.putText(frame, f"{label} {color_text}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Detección de prendas + color", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
