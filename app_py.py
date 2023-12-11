import streamlit as st
import tensorflow as tf
import h5py
import io
import zipfile

buffer = bytes()

# Open zip parts
zip_parts = [open(f'model.zip.{i:03d}', 'rb') for i in range(1, 43)]

# Read zip parts into buffer
for part in zip_parts:
    buffer += part.read()

# Create a ZipFile from the buffer
zf = zipfile.ZipFile(io.BytesIO(buffer))

# Check if 'model.h5' is present in the zip file
if 'model.h5' not in zf.namelist():
    raise FileNotFoundError("model.h5 not found in the zip archive")

# Read 'model.h5' from the zip file into an in-memory file object
f = io.BytesIO(zf.read('model.h5'))

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
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)

def import_and_predict(image_data, model):
    size = (224, 224)
    image_object = ImageOps.fit(image_data, size, Image.LANCZOS)
    image_array = np.asarray(image_object)
    image_cv = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    img_reshape = image_cv[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, load_model())
    class_names = ['Vegetables', 'Packages', 'Fruits']
    string = "This image is: " + class_names[np.argmax(predictions)]
    st.success(string)
