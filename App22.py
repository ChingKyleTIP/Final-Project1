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
        # Open the image using PIL
        image = Image.open(image_data)
        large_arr = np.fromfunction(lambda x, y, z: (x+y)//(z+1),
                            (256, 256, 3)).astype(np.uint8)
        large_img = PIL.Image.fromarray(large_arr)

        if image.mode != 'L':
            image = image.convert('L')


        img = np.asarray(image)
        img = img / 255.0
        img_reshape = img[np.newaxis, ..., np.newaxis]
        prediction = model.predict(img_reshape)
        return prediction
    except Exception as e:
        st.error(f"Error processing the image: {str(e)}")
        return None

if file is None:
    st.text("Please upload an image file.")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(file, model)
    class_names = ['marvel(1)',
                   'harry-potter(2)',
                   'star-wars(3)',
                   'jurassic-world(4)']
    result_string = "OUTPUT: " + class_names[np.argmax(prediction)]
    st.success(result_string)
