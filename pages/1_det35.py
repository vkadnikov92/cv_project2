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

# # Устанавливаем логгер для детектрона
# import detectron2

# Импорты
import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow

# # Зоопарк моделей (по аналогии с torchvision.models)
from detectron2 import model_zoo
# # Отдельный класс для предикта разными моделями
from detectron2.engine import DefaultPredictor
# # Всея конфиг: все будем делать через него
from detectron2.config import get_cfg

# from models.detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog, DatasetCatalog


# Функция для загрузки изображения из URL
def load_image_from_url(url):
    response = requests.get(url)
    img_data = BytesIO(response.content)
    return Image.open(img_data)

device = torch.device('cpu')

cfg = get_cfg()
cfg.MODEL.DEVICE = "cpu"
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "models/detectron2_500ep_weights.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
predictor = DefaultPredictor(cfg)

image = None
st.title("Сегментируем пляжные фото моделью семейства detectron2")

uploaded_file = st.file_uploader("Загрузите изображение в форматах 'jpg', 'png' или 'jpeg'", type=["jpg", "png", "jpeg"])
url = st.text_input("Или введите URL изображения...")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image.', use_column_width=True)
elif url:
    try:
        image = load_image_from_url(url).convert("RGB")
        st.image(image, caption='Loaded Image from URL.', use_column_width=True)
    except:
        st.write("Ошибка при загрузке изображения по URL. Проверьте ссылку и попробуйте еще раз.")

if image:
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)
    output_detectron = predictor(image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())
    segmented_img = output_detectron["instances"].pred_masks.cpu().numpy()

    if segmented_img.shape[0] > 0:  # Проверяем наличие масок
        combined_mask = np.max(segmented_img, axis=0)
        masked_image = np.array(image) * np.expand_dims(combined_mask, axis=2)
        masked_image_pil = Image.fromarray(masked_image.astype(np.uint8))
        st.image(masked_image_pil, caption='Image with Segmentation.', use_column_width=True)
    else:
        st.write("Объекты на изображении не обнаружены.")
else:
    st.write("Пожалуйста, загрузите изображение или предоставьте URL.")
