import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import torch
from torchvision.transforms import ToTensor
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import cv2
from detectron2.utils.visualizer import Visualizer

from detectron2.data import MetadataCatalog
import json


# Загрузка JSON-файла с метаданными
with open("models/_annotations.coco.json", "r") as json_file:
    sea_train_meta_data = json.load(json_file)

# Регистрация метаданных (если это не было сделано ранее)
if "sea_train" not in MetadataCatalog.list():
    MetadataCatalog.get("sea_train").set(thing_classes=[cat["name"] for cat in sea_train_meta_data["categories"]])


# Функция для загрузки изображения из URL
def load_image_from_url(url):
    response = requests.get(url)
    img_data = BytesIO(response.content)
    return Image.open(img_data)

# Загрузка модели detectron2
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "models/detectron2_500ep_weights.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cpu"
predictor = DefaultPredictor(cfg)


# Заголовок Streamlit
st.title("Сегментация изображения с помощью detectron2")

# Выбор источника изображения (файл или URL)
image_source = st.radio("Выберите источник изображения:", ("Файл", "URL"))

# Загрузка изображения
if image_source == "Файл":
    uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='Загруженное изображение', use_column_width=True)
else:
    url = st.text_input("Введите URL изображения...")
    if url:
        try:
            image = load_image_from_url(url)
            st.image(image, caption='Изображение из URL', use_column_width=True)
        except:
            st.write("Ошибка при загрузке изображения по URL. Проверьте ссылку и попробуйте еще раз.")

# Обработка изображения
if "image" in locals():
    try:
        # Загрузка изображения и выполнение сегментации
        image_nparray = np.array(image)
        outputs = predictor(image_nparray)

        # Визуализация результата сегментации с использованием зарегистрированных метаданных
        v = Visualizer(image_nparray[:, :, ::-1], metadata="sea_train", scale=0.8)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        st.image(out.get_image()[:, :, ::-1], caption='Результат сегментации', use_column_width=True)
    except Exception as e:
        st.write("Ошибка при обработке изображения:", e)
