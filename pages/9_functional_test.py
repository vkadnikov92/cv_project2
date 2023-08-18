import streamlit as st
import requests
import numpy as np
import cv2

import streamlit as st
from PIL import Image

# Импорты Detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer

import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from torchvision.transforms import ToTensor
import numpy as np
import torch
from models.autoencoder import ConvAutoencoder, load_model_with_weights
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import ToPILImage, Compose
from torchvision import transforms as T



# classes = ['sea', 'beach', 'sea', 'sky']

# cfg = get_cfg()
# yaml_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml" # путь к виду модели из библиотеки detectron2
# model_weights = "./models/beach_2500ep_weights.pth"  # путь к весам обученной нами модели
# cfg.merge_from_file(model_zoo.get_config_file(yaml_path))
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
# cfg.MODEL.WEIGHTS =  model_weights 
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
# cfg.MODEL.DEVICE = "cpu"

# # Создаем объект предиктора
# predictor = DefaultPredictor(cfg)





# cfg.merge_from_file(model_zoo.get_config_file(yaml_path))

# # Устанавливаем порог для детекции: если уровень доверия меньше порога, детекция не состоится
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
# # Загружаем модель
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(yaml_path)

# # Создаем объект предиктора
# predictor = DefaultPredictor(cfg)

# # Передаем в объект загруженное выше изображение
# outputs = predictor(im)


# # Заголовок Streamlit
# st.title("Загрузка и отображение изображения")

# # Поле ввода для URL изображения
# image_url = st.text_input("Введите URL изображения:")

# # Загрузка изображения по URL, если URL был введен
# if image_url:
#     response = requests.get(image_url)
#     if response.status_code == 200:
#         image_nparray = np.frombuffer(response.content, np.uint8)
#         im = cv2.imdecode(image_nparray, cv2.IMREAD_COLOR)
#         st.image(im, caption='Загруженное изображение', use_column_width=True)
#     else:
#         st.write("Ошибка при загрузке изображения. Пожалуйста, проверьте URL и попробуйте еще раз.")


# Заголовок Streamlit
st.title("Сегментация изображения с помощью detectron2")

# Загрузка изображения
uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "png", "jpeg"])

# Если изображение загружено
if uploaded_file:
    image = Image.open(uploaded_file)
    im = np.array(image)

    # Используем дефолтный конфиг
    cfg = get_cfg()
    classes = ['sea', 'beach', 'sea', 'sky']
    yaml_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml" # путь к виду модели из библиотеки detectron2
    model_weights = "./models/beach_2500ep_weights.pth"  # путь к весам обученной нами модели
    yaml_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(yaml_path))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    cfg.MODEL.WEIGHTS = model_weights  # Замените на путь к вашим весам модели
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
    cfg.MODEL.DEVICE = "cpu"

    # Создаем объект предиктора
    predictor = DefaultPredictor(cfg)

    # Выполняем сегментацию
    outputs = predictor(im)

    # Визуализация результата сегментации
    v = Visualizer(im[:, :, ::-1], metadata=None, scale=0.8)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    st.image(out.get_image()[:, :, ::-1], caption='Результат сегментации', use_column_width=True)