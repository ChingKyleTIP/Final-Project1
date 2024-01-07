import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps  # Add this line
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    model_path = 'weights.best.hdf5'
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model

def import_and_predict(image_data,model):
    size=(64,64)
    image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
    img=np.asarray(image)
    img_reshape=img[np.newaxis,...]
    prediction=model.predict(img_reshape)
    return prediction
if file is None:
    st.text("Please upload an image file")
    
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
            prediction = import_and_predict(image, model)
            if prediction is not None:
                class_names = ['harry-potter', 'starwars', 'jurassic-world', 'marvel']
                tring="OUTPUT : "+class_names[np.argmax(prediction)]
                st.success(string)

if __name__ == "__main__":
    main()
