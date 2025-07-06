# YOLOv8 - Detección de Prendas y Análisis de Color

Este proyecto utiliza **YOLOv8** para detectar personas y prendas (`top`, `bottom`, `exposed`) en imágenes o video, y luego analiza el **color dominante** de cada prenda detectada.

---

## 📁 Estructura del Proyecto

- `transfer_learning_yolo.ipynb`: Notebook en Colab para:
  - Descargar y filtrar dataset desde Roboflow
  - Renombrar clases y preparar etiquetas (`top`, `bottom`, `exposed`)
  - Entrenar modelo YOLOv8 personalizado
  - Exportar `.pt` entrenado

- `TGC_final_test.py`: Script para usar el modelo en video o webcam y visualizar etiquetas + color RGB

---

## ✅ Clases usadas

| Clase     | Incluye...                                   |
|-----------|----------------------------------------------|
| `top`     | Shirt, Jacket, Hoodie, SleevelessShirt, etc. |
| `bottom`  | Pants, Shorts, Skirt                         |
| `exposed` | Persona sin prenda visible                   |

---

## 🎥 Demo
 
https://github.com/user-attachments/assets/18b93f2e-9503-4253-b177-a69822a6f1c6

---

Modelo entrenado con YOLOv8n durante ~50 épocas usando un subconjunto personalizado extraído de Roboflow.
