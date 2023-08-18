import streamlit as st
import ssl

# Отключение проверки SSL-сертификата
ssl._create_default_https_context = ssl._create_unverified_context

st.set_page_config(
    page_title='Проект. Введение в нейронные сети',
    layout='wide'
)

st.sidebar.header("Home page")
c1, c2 = st.columns(2)
# c1 = st.columns(2)
# c2.image('neiro2.jpg')
c1.markdown("""
# CV project: multipage app
### 1. Сегментация продуктов и блюд с помощью двухэтапной или YOLOv8 моделей
### 2. Детекция ветрогенераторов с помощью YOLOv8 модели
### 3. Очищение документов от шумов с помощью автоэнкодера 
""")