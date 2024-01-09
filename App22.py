import streamlit as st
import tensorflow as tf
from PIL import Image
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

class_names = ['marvel(1)',
               'harry-potter(2)',
               'star-wars(3)',
               'jurassic-world(4)']

def import_and_predict(image_data, model, class_names):
    size = (64, 64)

    try:
        image = Image.open(image_data)
        # Resize the image to (64, 64) using TensorFlow function
        image = tf.image.resize(np.array(image), size)

        # Convert the image to grayscale using TensorFlow function
        image = tf.image.rgb_to_grayscale(image)

        img_array = tf.image.convert_image_dtype(image, tf.float32)
        img_array = tf.expand_dims(img_array, 0)
        
        # Reshape the image array to match the model input shape
        img_array = tf.image.resize_with_crop_or_pad(img_array, target_height=64, target_width=64)
        
        prediction = model.predict(img_array)
        
        prediction = np.squeeze(prediction)
        
        return prediction
    except Exception as e:
        st.error(f"Error processing the image: {str(e)}")
        return None

if file is None:
    st.text("Please upload an image file.")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(file, model, class_names)
    
    # Check if the prediction index is within the valid range
    if 0 <= np.argmax(prediction) < len(class_names):
        result_string = "OUTPUT: " + class_names[np.argmax(prediction)]
        st.success(result_string)
    else:
        st.error("Invalid prediction index.")
