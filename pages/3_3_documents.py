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


# Функция для загрузки изображения из URL
def load_image_from_url(url):
    response = requests.get(url)
    img_data = BytesIO(response.content)
    return Image.open(img_data)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = torch.device('cpu')

# Создание модели
model_autoencoder = ConvAutoencoder().to(device)
# Загрузка весов
weights_path = 'models/autoencoder_Salman.pt'
model_autoencoder = load_model_with_weights(model_autoencoder, weights_path)

model_autoencoder.eval()


image = None

# Заголовок Streamlit
st.title("Задача по очистке документов от 'шумов' силами автоэнкодера")

# Выбор источника изображения (файл или URL)
image_source = st.radio("Выберите источник изображения:", ("Файл", "URL"))

uploaded_file = None
url = None

if image_source == "Файл":
    uploaded_file = st.file_uploader("Загрузите изображение в форматах 'jpg', 'png' или 'jpeg'", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Image.', use_column_width=True)
elif image_source == "URL":
    url = st.text_input("Или введите URL изображения...")
    if url:
        try:
            image = load_image_from_url(url).convert("RGB")
            st.image(image, caption='Loaded Image from URL.', use_column_width=True)
        except:
            st.write("Ошибка при загрузке изображения по URL. Проверьте ссылку и попробуйте еще раз.")

if image: 

    desired_size = (540, 420)

    transform = T.Compose([
        # T.Resize(desired_size),
        # T.CenterCrop(299),
        # ToPILImage(),
        lambda x: x.convert('L'),  # Преобразование в черно-белое
        T.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    ])


    image_tensor = transform(image).unsqueeze(0)
    # image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).float() / 255
    output = model_autoencoder(image_tensor)
    cleaned_img_tensor = model_autoencoder(image_tensor)
    # cleaned_img = (cleaned_img_tensor.squeeze(0).squeeze(0) * 255).cpu().numpy().astype(np.uint8)
    cleaned_img = T.ToPILImage()(cleaned_img_tensor.squeeze(0))
    st.image(cleaned_img, caption='Clean Image.', use_column_width=True)
    # _, predicted_class = outputs.max(1)  # Получаем класс с максимальной вероятностью
    # class_name = labels[predicted_class.item()]
else:
    st.write("Пожалуйста, загрузите изображение или предоставьте URL.")