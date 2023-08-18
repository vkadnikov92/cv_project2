import streamlit as st
import requests
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import torch
from models.autoencoder import ConvAutoencoder, load_model_with_weights
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import ToPILImage, Compose
from torchvision import transforms as T
import json
# from detectron2.data.datasets import register_coco_instances

# Импорты Detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


# Заголовок Streamlit
st.title("Задача сегментации")
st.title("Ищем пляжи, море и небо силами Detectron2")

# Выбор источника изображения (файл или URL)
image_source = st.radio("Выберите источник изображения:", ("Файл", "URL"))

# Загрузка изображения
if image_source == "Файл":
    uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
else:
    url = st.text_input("Введите URL изображения...")
    if url:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))

# Если изображение загружено
if 'image' in locals():
    im = np.array(image)

    # Используем дефолтный конфиг
    cfg = get_cfg()
    classes = ['sea', 'beach', 'sea', 'sky']
    yaml_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml" # путь к виду модели из библиотеки detectron2
    model_weights = './models/beach_2500ep_weights.pth'   # путь к весам обученной нами модели
    cfg.merge_from_file(model_zoo.get_config_file(yaml_path))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    cfg.MODEL.WEIGHTS = model_weights  # Замените на путь к вашим весам модели
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
    cfg.MODEL.DEVICE = "cpu"

    # Создаем объект предиктора
    predictor = DefaultPredictor(cfg)
    # Выполняем сегментацию
    outputs = predictor(im[:, :, ::-1])

    # # Загрузка JSON-файла с метаданными
    # with open("./models/_annotations.coco.json", "r") as json_file:
    #     metadata = json.load(json_file)

    # Отображаем исходное изображение
    st.image(image, caption='Загруженное изображение.', use_column_width=True)

    # Визуализация результата сегментации
    v = Visualizer(im[:, :, ::-1], metadata=None, scale=0.8)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    st.image(out.get_image()[:, :, ::-1], caption='Результат сегментации', use_column_width=True)