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

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device('cpu')


cfg = get_cfg()
cfg.MODEL.DEVICE = "cpu"
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

# Указываете путь к вашему файлу с весами
cfg.MODEL.WEIGHTS = "models/detectron2_500ep_weights.pth"

# Здесь можно установить порог обнаружения или другие параметры, если это необходимо
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

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

if image:  # Если изображение успешно загружено (будь то через файл или URL)
    
    desired_size = (540, 420)

    transform = T.Compose([
        # T.Resize(desired_size),
        # T.CenterCrop(299),
        # ToPILImage(),
        # lambda x: x.convert('L'),  # Преобразование в черно-белое
        T.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    ])

    image_tensor = transform(image).unsqueeze(0)
    # image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).float() / 255

    # output = predictor(image_tensor)
    output_detectron = predictor(image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())
    # output_detectron = predictor(image_tensor)
    segmented_img = output_detectron["instances"].pred_masks  # получение маски сегментации

    # Применение сегментированной маски к изображению (cleaned_img_tensor)
    cleaned_img_with_segmentation = output_detectron * segmented_img

    # Преобразование тензора в изображение PIL
    cleaned_img_pil_with_segmentation = T.ToPILImage()(cleaned_img_with_segmentation.squeeze(0))

    # Вывод результата с сегментацией
    st.image(cleaned_img_pil_with_segmentation, caption='Clean Image with Segmentation.', use_column_width=True)
else:
    st.write("Пожалуйста, загрузите изображение или предоставьте URL.")