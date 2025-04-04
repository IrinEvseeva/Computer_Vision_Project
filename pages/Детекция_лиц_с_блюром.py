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


st.set_page_config(
    layout="wide",
    page_title="Детекция лиц с последующей маскировкой детектированной области",
    page_icon="👤",
)
@st.cache_resource
def load_model():
    model = YOLO('/home/savely/ds_bootcamp/ds-phase-2/Computer-vision-project/model/face_detection.pt')  
    return model

model = load_model()

def process_image(image):
    # Преобразуем изображение в формат, подходящий для модели
    results = model(image, conf=0.4)
    annotated_frame = np.array(image.copy())

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Извлекаем область изображения
            roi = annotated_frame[y1:y2, x1:x2]
            # Преобразуем область в объект PIL Image
            roi_pil = Image.fromarray(roi)
            # Применяем размытие с помощью PIL
            blurred_roi_pil = roi_pil.filter(ImageFilter.GaussianBlur(radius=15))
            # Преобразуем размытую область обратно в массив NumPy
            blurred_roi = np.array(blurred_roi_pil)
            # Заменяем оригинальную область на размытую
            annotated_frame[y1:y2, x1:x2] = blurred_roi

    # Преобразуем измененный массив обратно в объект PIL Image
    processed_image = Image.fromarray(annotated_frame)
    return processed_image


st.subheader('Модель, для детекции лиц на фото')

st.write('**Вы можете загрузить одно или несколько фотографий, а также загурзить фото по URL ссылке из интернета. Модель YOLO12 будет блюрит лица на фото**')

uploaded_files = st.file_uploader("Загрузите изображения", type=["jpg", "jpeg", "png", "bmp", "tiff"], accept_multiple_files=True)
image_url = st.text_input('Или вставьте ссылку на изображение')

image = None

if uploaded_files or image_url:
    st.subheader('Результаты обработки')

    if uploaded_files:
        st.write('**Загруженные изображения:**')
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            st.image(image, caption='Загруженное изображение', use_container_width=True)
            processed_image = process_image(image)
            st.image(processed_image, caption='Обработанное изображение', use_container_width=True)

    if image_url:
        try:
            response = requests.get(image_url)
            image = Image.open(io.BytesIO(response.content))
            st.image(image, caption='Загруженное изображение', use_container_width=True)
            processed_image = process_image(image)
            st.image(processed_image, caption='Обработанное изображение', use_container_width=True)
        except Exception as e:
            st.error(f"Ошибка при загрузке изображения по ссылке: {e}")