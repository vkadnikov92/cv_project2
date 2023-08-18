import io
import streamlit as st
from PIL import Image, ImageDraw
from ultralytics import YOLO  # Импортируем класс YOLO из ultralytics

# Заголовок Streamlit
st.title("Задача сегментации")
st.title("Ищем пляжи, море и небо силами YOLO8")


# Выбор источника изображения (файл или URL)
image_source = st.radio("Выберите источник изображения:", ("Файл", "URL"))

# Загрузка изображения
if image_source == "Файл":
    uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        pil_image = Image.open(uploaded_file)
else:
    url = st.text_input("Введите URL изображения...")
    if url:
        response = requests.get(url)
        pil_image = Image.open(BytesIO(response.content))

# Если изображение загружено
if 'pil_image' in locals():
    # Отобразить изображение
    st.image(pil_image, caption="Загруженное изображение", use_column_width=True)

    # Загрузить предобученную модель YOLO
    model = YOLO('models/beach2.pt')  # Укажите путь к предобученной модели или имя (например, 'yolov5s')

    # Выполнить инференс на загруженном изображении
    results = model.predict(pil_image, imgsz=320, conf=0.5)

    from PIL import Image

        # Show the results
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
     
    # Отобразить аннотированное изображение
        st.image(im, caption="Predicted Image", use_column_width=True)

    