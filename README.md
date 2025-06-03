# CatDogAI

CatDogAI es un proyecto de inteligencia artificial que clasifica imágenes de perros y gatos usando una red neuronal convolucional (CNN). Incluye una interfaz gráfica sencilla para visualizar cada imagen junto con la probabilidad estimada de que sea un gato o un perro.

---

## Características

   - Clasificación binaria de imágenes: gato vs perro.
   - Dataset personalizado para imágenes mezcladas en una sola carpeta (sin subcarpetas).
   - Entrenamiento con PyTorch usando CNN simple pero eficaz.
   - Interfaz gráfica desarrollada con Tkinter para mostrar imágenes y probabilidades.
   - Compatible con CPU y GPU (si está disponible).

---

## Instalación

1. Clona este repositorio:

   ```bash
   git clone https://github.com/tuusuario/CatDogAI.git
   cd CatDogAI

2. Instala las dependencias:
   
   ```bash
   pip install -r requirements.txt

3. Prepara tu carpeta ./train con imágenes de gatos y perros. Los archivos deben comenzar con:
   - gato para imágenes de gatos.
   - perro para imágenes de perros.

## Uso
Ejecuta el script principal para entrenar el modelo y abrir la interfaz gráfica:
   ```bash
   python main.py
   ```
La ventana mostrará las imágenes junto con las probabilidades de que cada una sea gato o perro. Puedes navegar entre las imágenes usando los botones Anterior y Siguiente.

## Modelo
Se utiliza una CNN con dos capas convolucionales seguidas de capas lineales para clasificación binaria. El entrenamiento es por 5 épocas con Adam optimizer y función de pérdida CrossEntropy.
