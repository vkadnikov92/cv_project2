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

# Ваш список классов
classes = ["class1", "class2", "class3"] 

def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
    cfg.MODEL.WEIGHTS = "./models/beach_2500ep_weights.pth"  #  на путь к весам
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50
    cfg.MODEL.DEVICE = "cpu"

    return cfg

def predict_and_visualize(image, cfg):
    predictor = DefaultPredictor(cfg)
    outputs = predictor(image)
    sea_valid_meta = MetadataCatalog.get("sea_valid")
    v = Visualizer(image[:, :, ::-1], metadata=sea_valid_meta, scale=0.8)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    return out.get_image()[:, :, ::-1]

def main():
    st.title("Загрузите ваше изображение")
    
    uploaded_file = st.file_uploader("Выберите файл", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        image = np.array(image)
        
        # Отображаем исходное изображение
        st.image(image, caption='Загруженное изображение.', use_column_width=True)
        
        # Прогоняем изображение через модель и отображаем результат
        cfg = setup_cfg()
        result_img = predict_and_visualize(image, cfg)
        st.image(result_img, caption='Результат после обработки моделью.', use_column_width=True)

if __name__ == "__main__":
    main()