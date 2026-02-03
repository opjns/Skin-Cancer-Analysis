import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = (224, 224)
THRESHOLD = 0.4

st.set_page_config(
    page_title="Skin Cancer Detection",
    layout="centered"
)

st.title("Skin Cancer Detection using")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("skin_cancer_cnn.keras")

model = load_model()

def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    image = np.array(image)

    if image.shape[-1] == 4:
        image = image[:, :, :3]

    image = image / 255.0
    return np.expand_dims(image, axis=0)

uploaded_file = st.file_uploader(
    "Upload a skin lesion image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)

    x = preprocess_image(image)
    pred = model.predict(x)[0][0]

    if pred > THRESHOLD:
        st.error("⚠️ Malignant")
    else:
        st.success("✅ Benign")

    st.write(f"Confidence: {pred:.2f}")
