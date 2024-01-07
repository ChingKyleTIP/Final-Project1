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

    try:
        # Open the image using PIL
        image = Image.open(image_data)

        # Convert the image to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')

        # Resize the image
        image = ImageOps.fit(image, size, Image.ANTIALIAS)

        # Convert the image to a numpy array
        img = np.asarray(image)

        # Normalize the pixel values to be between 0 and 1
        img = img / 255.0

        # Reshape the image for model input
        img_reshape = img[np.newaxis, ..., np.newaxis]

        # Make predictions using the loaded model
        prediction = model.predict(img_reshape)

        return prediction

    except Exception as e:
        st.error(f"Error processing the image: {str(e)}")
        return None

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
