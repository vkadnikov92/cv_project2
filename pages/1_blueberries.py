import io
import streamlit as st
from PIL import Image, ImageDraw
from ultralytics import YOLO  # Импортируем класс YOLO из ultralytics

# Заголовок приложения
st.title("Image Processing App")

# Загрузка изображения с помощью Streamlit
uploaded_image = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

# Если изображение загружено
if uploaded_image is not None:
    # Отобразить изображение
    st.image(uploaded_image, caption="Загруженное изображение", use_column_width=True)

    # Создать PIL объект из загруженного изображения
    pil_image = Image.open(uploaded_image)

    # Загрузить предобученную модель YOLO
    model = YOLO('model/best.pt')  # Укажите путь к предобученной модели или имя (например, 'yolov5s')

    # Выполнить инференс на загруженном изображении
    results = model.predict(pil_image, imgsz=320, conf=0.5)

    from PIL import Image

        # Show the results
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
     
    # Отобразить аннотированное изображение
        st.image(im, caption="Predicted Image", use_column_width=True)