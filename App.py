import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('best_model.hdf5')
    return model

def import_and_predict(image_data, model):
    size = (64, 64)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

def main():
    st.write("# World of Lego toys classifier")

    file = st.file_uploader("Upload a Lego model from your computer", type=["jpg", "png"])

    if file is None:
        st.text("Please upload an image file")
    else:
        model = load_model()
        image = Image.open(file)
        st.image(image, use_column_width=True)
        prediction = import_and_predict(image, model)
        class_names = ['harry-potter', 'starwars', 'jurassic-world', 'marvel']
        output_class = class_names[np.argmax(prediction)]
        st.success(f"OUTPUT: {output_class}")

if __name__ == "__main__":
    main()
