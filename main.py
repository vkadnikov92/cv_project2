import streamlit as st
import ssl
from PIL import Image
import requests
from io import BytesIO

# Отключение проверки SSL-сертификата
ssl._create_default_https_context = ssl._create_unverified_context

# Функция для загрузки изображения из URL
def load_image_from_url(url):
    response = requests.get(url)
    img_data = BytesIO(response.content)
    return Image.open(img_data)

# URL изображения
url = 'https://sun9-70.userapi.com/impf/c857636/v857636932/99a20/ePl5azpmotw.jpg?size=603x800&quality=96&sign=4931e8bd4232808e47fd823ece43b6c4&c_uniq_tag=pPqbgunZrFzgPkKHuJFM-5BMRGa5yiE4nZ-TjR19E6c&type=album'
# url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ35F87bVKuhwVypa_ZIlW9ptbdVQOAhm5UNQ&usqp=CAU'

# Загрузка изображения
image = load_image_from_url(url).convert("RGB")

# Ресайз изображения справа
image_resized = image #image.resize((300, 400))


# Настройка страницы
st.set_page_config(
    page_title='Проект. Введение в нейронные сети',
    layout='wide'
)

# Вывод боковой панели
st.sidebar.header("Home page")
c1, c2 = st.columns((1, 2))  # Используйте columns с кортежем для задания ширины колонок

# Вывод изображения
c2.image(image_resized, caption='Loaded Image from URL (Resized).', use_column_width=True)

# Вывод текста и заголовка в первой колонке
c1.markdown("""
# CV project: multipage app
### 1. Сегментация продуктов и блюд с помощью двухэтапной или YOLOv8 моделей
### 2. Детекция ветрогенераторов с помощью YOLOv8 модели
### 3. Очищение документов от шумов с помощью автоэнкодера 
""")