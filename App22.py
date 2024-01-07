import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Function to load the pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('weights.best.hdf5')
    return model

# Load the pre-trained model
model = load_model()

# Streamlit UI
st.write("""
# Welcome to the Lego World of Toys
""")

# File uploader for image selection
file = st.file_uploader("Choose a toy photo from your computer", type=["jpg", "png"])

# Function to preprocess the image and make predictions
def import_and_predict(image_data, model):
    size = (64, 64)
    # Resize and preprocess the image
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    # Make predictions using the loaded model
    prediction = model.predict(img_reshape)
    return prediction

# Check if a file is uploaded
if file is None:
    st.text("Please upload an image file.")
else:
    # Display the uploaded image
    image = Image.open(file)
    st.image(image, use_column_width=True)
    
    # Make predictions and display the result
    prediction = import_and_predict(image, model)
    
    # Define class names
    class_names = ['marvel(1)', 'harry-potter(2)', 'star-wars(3)', 'jurassic-world(4)']
    
    # Create the result string
    result_string = "OUTPUT: " + class_names[np.argmax(prediction)]
    
    # Display the result
    st.success(result_string)
