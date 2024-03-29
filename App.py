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

def import_and_predict(image_data, model):
    size = (64, 64)

    try:

        image = Image.open(image_data)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        img = np.asarray(image)
        img_reshape = img[np.newaxis, ...]
        prediction = model.predict(img_reshape)

        return prediction
    except Exception as e:
        return None


if file is not None:
    st.image(file, use_column_width=True)
    if st.button("Predict"):
        try:
            prediction = import_and_predict(file, model)
            if prediction is not None:
                st.write("Raw Prediction Output:")
                st.write(prediction)
                
                class_names = ['marvel(1)', 'harry-potter(2)', 'star-wars(3)', 'jurassic-world(4)']
                result_class = class_names[np.argmax(prediction)]
                st.success(f"Predicted Class: {result_class}")
                
            else:
                st.write(prediction)
                class_names = ['marvel(1)', 'harry-potter(2)', 'star-wars(3)', 'jurassic-world(4)']
                result_class = class_names[np.argmax(prediction)]
                st.success(f"Predicted Class: {result_class}") 
                
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
