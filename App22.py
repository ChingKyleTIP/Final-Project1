import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('weights.best.hdf5')
    return model

model = load_model()

st.write("""
# Welcome to the World of Lego Toys
""")

file = st.file_uploader("Choose a toy photo from your computer", type=["jpg", "png"])

if file is not None:
    st.image(file, use_column_width=True)
    if st.button("Predict"):
        prediction = import_and_predict(file, model)
        class_names = ['marvel(1)', 'harry-potter(2)', 'star-wars(3)', 'jurassic-world(4)']
        if prediction is not None:
            string = "OUTPUT: " + class_names[np.argmax(prediction)]
            st.success(string)
        else:
            st.error("Error processing the image.")
