import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image, ImageFilter
import requests
from io import BytesIO
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from ultralytics import YOLO
import io
import urllib.request
import cv2

# Настройки страницы должны быть в начале
st.set_page_config(
    layout="wide",
    page_title="Детекция ветрогенераторов (Модель YOLO12)",
    page_icon="💨",
)


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Инициализация модели
@st.cache_resource
def load_model():
    model = YOLO('/home/savely/ds_bootcamp/ds-phase-2/Computer-vision-project/model/wind_turbines.pt')  
    return model

# Загружаем модель один раз при старте приложения
model = load_model()

def get_prediction(img, model) -> int:
    start = time.time()
    results = model.predict(img) # Получаем вывод модели
    end = time.time()
    pred_images = [result.plot() for result in results]
    return end-start, pred_images


# Функция предсказания
def get_prediction(img, model) -> tuple:
    start = time.time()
    
    # Если изображение уже PIL Image, конвертируем в numpy array
    if isinstance(img, Image.Image):
        img_np = np.array(img)
        # Конвертируем RGB в BGR если нужно (для OpenCV)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    else:
        img_np = img
    
    results = model.predict(img_np)
    end = time.time()
    
    pred_images = []
    for result in results:
        plotted_img = result.plot()
        
        # Конвертация цветового пространства
        rgb_img = cv2.cvtColor(plotted_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        pred_images.append(pil_img)
    
    return end-start, pred_images

if 'predictions' not in st.session_state:
    st.session_state.predictions = []


# Streamlit-интерфейс
st.subheader('Модель, для детекции лиц на фото')

st.write('**Вы можете загрузить одно или несколько фотографий, а также загурзить фото по URL ссылке из интернета. Модель YOLO12 будет детектировать изображение ветренных мельниц**')


uploaded_file = st.sidebar.file_uploader(
    label='Загружайте снимок сюда:', 
    type=['jpeg', 'png', 'jpg'], 
    accept_multiple_files=True
)

if uploaded_file is not None:
    for file in uploaded_file:
        image = Image.open(file)
        if model is not None:
            sec, results = get_prediction(image, model)
            st.write(f'''Время выполнения предсказания: __{sec:.4f} секунды__ 
        \nРезультат детекции:''')
            st.image(results, use_container_width=True)

link = st.sidebar.text_input(label='Вставьте сюда ссылку на снимок')
if link is not '':
    image = Image.open(urllib.request.urlopen(link))
    if model is not None:
        sec, results = get_prediction(image, model)
        st.write(f'''Время выполнения предсказания: __{sec:.4f} секунды__ 
        \nРезультат детекции:''')
        st.image(results, use_container_width=True)


# Streamlit-интерфейс
# uploaded_files = st.file_uploader("Загрузите изображение клеток крови для определения их типа", 
#                                 type=["jpg", "png", "jpeg"], 
#                                 accept_multiple_files=True)  # Разрешаем множественную загрузку

# image_url = st.text_input("Загрузи фото по ссылке")

# if image_url:
#     response = requests.get(image_url)
#     image = Image.open(BytesIO(response.content))
    
#     # Отображение изображения
#     st.image(image, caption="Загруженное изображение", use_container_width=True)

#     st.subheader("Результат")
#     label, elapsed = get_prediction(image, model)
#     st.write(f" **Предсказанный класс:** {label}")
#     st.caption(f"⏱ Время инференса: {elapsed:.3f} сек")

# if uploaded_files:  # uploaded_files - это список файлов
#     for uploaded_file in uploaded_files:  # Обрабатываем каждый файл отдельно
#         image = Image.open(uploaded_file).convert("RGB")  # Теперь uploaded_file - один файл
#         st.image(image, caption=f"Загруженное изображение: {uploaded_file.name}", use_container_width=True)
        
#         label, elapsed = get_prediction(image, model)
#         st.write(f"**Предсказанный класс:** {label}")
#         st.caption(f"⏱ Время инференса: {elapsed:.3f} сек")
#         st.markdown("---")  # Разделитель между результатами