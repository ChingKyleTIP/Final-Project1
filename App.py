import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('weights.best.hdf5')
    return model

def import_and_predict(image_data, model):
    size = (64, 64)
    
    try:
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    except AttributeError:
        st.error("Error processing the image. Please try again with a different image.")
        return None
    
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    
    try:
        prediction = model.predict(img_reshape)
        return prediction
    except Exception as e:
        st.error(f"Error making predictions: {e}")
        return None

def main():
    st.write("# Welcome to the World of Lego Toys")

    file = st.file_uploader("Choose a Lego toy photo from your computer", type=["jpg", "png"])

    if file is None:
        st.text("Please upload an image file")
    else:
        model = load_model()

        if model is not None:
            image = Image.open(file)
            st.image(image, use_column_width=True)

            # Debug: Print raw predictions
            prediction = import_and_predict(image, model)
            st.write("Raw Predictions:", prediction)

            if prediction is not None:
                class_names = ['harry-potter', 'jurassic-world', 'marvel', 'star-wars']
                output_class = class_names[np.argmax(prediction)]
                st.success(f"OUTPUT: {output_class}")

if __name__ == "__main__":
    main()
