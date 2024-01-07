import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps  # Add this line
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    model_path = 'best_model.hdf5'
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model

def main():
    st.write("# World of Lego toys classifier")

    file = st.file_uploader("Upload a Lego model from your computer", type=["jpg", "png"])

    if file is None:
        st.text("Please upload an image file")
    else:
        model = load_model()
        if model is not None:
            image = Image.open(file)
            st.image(image, use_column_width=True)
            
            # Debug: Print some information
            st.write(f"Image Size: {image.size}")
            st.write(f"Image Mode: {image.mode}")

            prediction = import_and_predict(image, model)
            if prediction is not None:
                st.write("Predictions:", prediction)

                class_names = ['harry-potter', 'starwars', 'jurassic-world', 'marvel']
                output_class = class_names[np.argmax(prediction)]
                st.success(f"OUTPUT: {output_class}")

if __name__ == "__main__":
    main()


