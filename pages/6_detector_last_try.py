import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from torchvision.transforms import ToTensor
import numpy as np
import torch
import torchvision.models as models
from torchvision.transforms import ToPILImage, Compose
from torchvision import transforms as T
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

import cv2
from detectron2.utils.visualizer import Visualizer


# Функция для загрузки изображения из URL
def load_image_from_url(url):
    response = requests.get(url)
    img_data = BytesIO(response.content)
    return Image.open(img_data)

# Установка устройства
device = torch.device('cpu')

# Функция для загрузки изображения из URL и декодирования в массив NumPy
def load_image_from_url(url):
    response = requests.get(url)
    image_nparray = np.frombuffer(response.content, np.uint8)
    image = cv2.imdecode(image_nparray, cv2.IMREAD_COLOR)
    return image

# Загрузка модели detectron2
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "models/detectron2_500ep_weights.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cpu"
predictor = DefaultPredictor(cfg)



# Заголовок Streamlit
st.title("Сегментация изображения с помощью detectron2")

# Введите URL изображения
url = st.text_input("Введите URL изображения...")

# Обработка изображения
if url:
    try:
        # Загрузка изображения и выполнение сегментации
        image = load_image_from_url(url)
        outputs = predictor(image)

        # Визуализация результата сегментации
        v = Visualizer(image[:, :, ::-1], metadata=None, scale=0.8)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        st.image(out.get_image()[:, :, ::-1], caption='Результат сегментации', use_column_width=True)
    except Exception as e:
        st.write("Ошибка при обработке изображения:", e)
else:
    st.write("Введите URL изображения выше.")


# # Загрузка модели detectron2
# cfg = get_cfg()
# cfg.MODEL.DEVICE = "cpu"
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.MODEL.WEIGHTS = "models/detectron2_500ep_weights.pth"
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
# predictor = DefaultPredictor(cfg)

# # Заголовок Streamlit
# st.title("Сегментируем пляжные фото моделью семейства detectron2")

# # Загрузка изображения
# uploaded_file = st.file_uploader("Загрузите изображение в форматах 'jpg', 'png' или 'jpeg'", type=["jpg", "png", "jpeg"])
# url = st.text_input("Или введите URL изображения...")

# # Обработка изображения
# if uploaded_file:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption='Uploaded Image.', use_column_width=True)
# elif url:
#     try:
#         image = load_image_from_url(url).convert("RGB")
#         st.image(image, caption='Loaded Image from URL.', use_column_width=True)
#     except:
#         st.write("Ошибка при загрузке изображения по URL. Проверьте ссылку и попробуйте еще раз.")

# if image:  # Если изображение успешно загружено (будь то через файл или URL)
#     transform = T.Compose([
#         T.ToTensor(),
#     ])
    
#     image_tensor = transform(image).unsqueeze(0)
    
#     output_detectron = predictor(image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())
#     segmented_masks = output_detectron["instances"].pred_masks  # получение масок сегментации
    
#     # Вывод всех масок сегментации
#     for i, mask in enumerate(segmented_masks):
#         st.image(mask, caption=f'Segmented Mask {i+1}', use_column_width=True)
# else:
#     st.write("Пожалуйста, загрузите изображение или предоставьте URL.")
