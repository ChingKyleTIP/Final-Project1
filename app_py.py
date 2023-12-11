import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
from tensorflow.keras import callbacks,optimizers
import h5py

@st.cache(allow_output_mutation=True)
def load_model():
    h5 = h5py.File(f)
    return tf.keras.models.load_model(h5)

st.write("""
         # Item Purchase
         """
        )

file = st.file_uploader("Choose item images from computer", type=["jpg", "png"])

import cv2
from PIL import Image,ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):
    size = (224, 224)    
    image_object = ImageOps.fit(image_data, size, Image.LANCZOS)
    image_array = np.asarray(image_object)
    image_cv = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    img_reshape = image_cv[np.newaxis,...]
    prediction=model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, load_model())
    class_names = ['Vegetables','Packages', 'Fruits']
    string="This image is: "+class_names[np.argmax(predictions)]
    st.success(string)
