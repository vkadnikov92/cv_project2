import requests
import numpy as np
import cv2
from io import BytesIO
import torch
from models.autoencoder import ConvAutoencoder, load_model_with_weights
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import ToPILImage, Compose
from torchvision import transforms as T
import json
from detectron2.data.datasets import register_coco_instances
import io
from PIL import Image, ImageDraw
from ultralytics import YOLO  # Импортируем класс YOLO из ultralytics

import streamlit as st


# Заголовок Streamlit
st.title("Задача сегментации")
st.title("Ищем голубикy силами YOLO8")

# Выбор источника изображения (файл или URL)
image_source = st.radio("Выберите источник изображения:", ("Файл", "URL"))

# Загрузка изображения
if image_source == "Файл":
    uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        pil_image = Image.open(uploaded_file)
else:
    url = st.text_input("Введите URL изображения...")
    if url:
        response = requests.get(url)
        pil_image = Image.open(BytesIO(response.content))

# Если изображение загружено
if 'pil_image' in locals():
    # Отобразить изображение
    st.image(pil_image, caption="Загруженное изображение", use_column_width=True)

    # Загрузить предобученную модель YOLO
    model = YOLO('./models/best.pt')  # Укажите путь к предобученной модели или имя (например, 'yolov5s')

    # Выполнить инференс на загруженном изображении
    results = model.predict(pil_image, imgsz=320, conf=0.5)

    # Отобразить результаты
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
     
        # Отобразить аннотированное изображение
        st.image(im, caption="Predicted Image", use_column_width=True)