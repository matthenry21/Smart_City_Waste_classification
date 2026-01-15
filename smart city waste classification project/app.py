import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --------------------
# Load models (cached)
# --------------------
@st.cache_resource
def load_models():
    binary = tf.keras.models.load_model("models/binary_waste_classifier.h5")
    multi  = tf.keras.models.load_model("models/recyclable_multiclass_model_best.h5")
    return binary, multi

binary_model, multi_model = load_models()

# --------------------
# Labels
# --------------------
binary_labels = {0: "Non-Recyclable", 1: "Recyclable"}
multi_labels = ["cardboard", "e-waste", "glass", "metal", "paper", "plastic"]

# --------------------
# Image preprocessing
# --------------------
def preprocess(image):
    image = image.resize((224,224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# --------------------
# Streamlit UI
# --------------------
st.title("♻️ Smart Waste Classification System")
st.write("Upload a waste image to classify it.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = preprocess(image)

    # Binary prediction
    bin_prob = binary_model.predict(img_array)[0][0]
    bin_pred = int(bin_prob > 0.5)

    if bin_pred == 0:
        st.error("🗑️ Non-Recyclable Waste")
    else:
        st.success("♻️ Recyclable Waste")

        # Multi-class prediction
        preds = multi_model.predict(img_array)
        class_idx = np.argmax(preds[0])
        label = multi_labels[class_idx]
        confidence = preds[0][class_idx] * 100

        st.write(f"### 🏷️ Type: **{label}**")
        st.write(f"Confidence: **{confidence:.2f}%**")
