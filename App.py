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
