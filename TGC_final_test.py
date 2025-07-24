import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# --- Cargar modelos ---
person_model = YOLO("models/model_final_v1.pt")      # Solo detecta 'person'
pose_model = YOLO("models/yolov8n-pose.pt")          # Pose estimation

# --- Cargar diccionario de colores ---
df_colors = pd.read_csv("color_dictionary.csv")
color_vectors = df_colors[["R", "G", "B"]].values

def match_color(rgb_tuple):
    closest_idx, _ = pairwise_distances_argmin_min([rgb_tuple], color_vectors)
    row = df_colors.iloc[closest_idx[0]]
    return row["name"], (int(row["R"]), int(row["G"]), int(row["B"]))

def get_dominant_color(image, crop_type='top'):
    h, w, _ = image.shape

    # Recorte horizontal: centro
    if crop_type == 'top':
        margin_x = int(w * 0.17)  # 66% central
    else:
        margin_x = int(w * 0.20)  # 60% central
    crop = image[:, margin_x:w - margin_x]

    if crop.size == 0:
        return (0, 0, 0)

    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pixels = crop.reshape(-1, 3)

    kmeans = KMeans(n_clusters=1, n_init=10)
    kmeans.fit(pixels)
    dominant_color = tuple(map(int, kmeans.cluster_centers_[0]))
    return dominant_color

# --- Abrir video o webcam ---
cap = cv2.VideoCapture("videos/street.mp4")
#cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = person_model(frame, iou=0.5, conf=0.5)[0]

    for box, cls in zip(detections.boxes.xyxy, detections.boxes.cls):
        if int(cls) != 0:
            continue

        x1, y1, x2, y2 = map(int, box)
        person_crop = frame[y1:y2, x1:x2].copy()

        pose_results = pose_model(person_crop, conf=0.3)[0]

        if pose_results.keypoints is not None and len(pose_results.keypoints.xy) > 0:
            keypoints = pose_results.keypoints.xy[0]

            try:
                l_shoulder = keypoints[5]
                r_shoulder = keypoints[6]
                l_hip = keypoints[11]
                r_hip = keypoints[12]

                top_y = int(min(l_shoulder[1], r_shoulder[1])) + y1
                bottom_y = int(max(l_hip[1], r_hip[1])) + y1

                # --- TOP ---
                top_roi = frame[top_y:bottom_y, x1:x2]
                top_color = get_dominant_color(top_roi, crop_type='top')
                top_name, top_rgb = match_color(top_color)

                cv2.rectangle(frame, (x1, top_y), (x2, bottom_y), top_rgb, 2)
                cv2.putText(frame, f"TOP: {top_name}", (x1, top_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, top_rgb, 2)

                # --- BOTTOM ---
                bottom_roi = frame[bottom_y:y2, x1:x2]
                bottom_color = get_dominant_color(bottom_roi, crop_type='bottom')
                bottom_name, bottom_rgb = match_color(bottom_color)

                cv2.rectangle(frame, (x1, bottom_y), (x2, y2), bottom_rgb, 2)
                cv2.putText(frame, f"BOTTOM: {bottom_name}", (x1, y2 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, bottom_rgb, 2)

            except IndexError:
                pass

        # Dibuja caja completa de la persona
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, "person", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Person + Top/Bottom Color", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
