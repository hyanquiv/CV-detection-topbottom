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



---

Modelo entrenado con YOLOv8n durante ~50 épocas usando un subconjunto personalizado extraído de Roboflow.
