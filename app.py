import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json

st.set_page_config(page_title="Smart Waste Classifier", page_icon="♻️")
st.title("♻️ Smart Waste Classification System")
st.write("CODE QUEST 2026 - AI Application (FCIT)")

@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model('waste_classifier_model.h5')
    with open('class_names.json', 'r') as f:
        classes = json.load(f)
    return model, classes

model, class_names = load_my_model()

uploaded_file = st.file_uploader("Choose a waste image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    st.write("Classifying... Please wait.")

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0

    if img_array.shape[-1] != 3:
        img_array = np.stack((img_array,)*3, axis=-1)

    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    best_class_idx = np.argmax(predictions)
    predicted_class = class_names[best_class_idx]
    confidence = predictions[best_class_idx] * 100

    st.success(f"**Prediction:** {predicted_class}")
    st.info(f"**Confidence:** {confidence:.2f}%")

    st.write("### Prediction Probabilities for All Classes:")
    chart_data = {class_names[i]: float(predictions[i]) for i in range(len(class_names))}
    st.bar_chart(chart_data)
