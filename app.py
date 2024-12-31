import streamlit as st
import keras
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

model = keras.models.load_model('model.keras')
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
st.set_page_config(layout='wide', page_title='Brain Tumor predictor')
st.write('# Brain tumor predictor :brain:')
st.write('## Brain tumor detector from cranial tomography images, using convolutional neural networks (CNN) '
         'architecture.')
st.sidebar.write('## Upload the image :gear:')
upload_file = st.sidebar.file_uploader('Upload an image', type='jpg')


def preprocess_img_and_predict(img_data, model):
    size = (300, 300)
    img = ImageOps.fit(img_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(img)
    img_reshape = img[np.newaxis, ...]
    print(img_reshape.shape)
    pred = model.predict(img_reshape)
    return pred


if upload_file is not None:
    if upload_file.size > MAX_FILE_SIZE:
        st.error('This file is very large, please upload a file smaller than  5 MB')
    else:
        img = Image.open(upload_file)
        st.write('## Asking for the model about the following image... :camera:')
        st.image(image=img, use_container_width=True)
        pred = preprocess_img_and_predict(img, model)
        print(pred)

        class_names = ['Glioma tumor', 'Meningioma tumor', 'No tumor', 'Pituitary tumor']
        print(np.argmax(pred))
        print(class_names[np.argmax(pred)])
        string = 'Detected Disease: ' + class_names[np.argmax(pred)]
        if class_names[np.argmax(pred)] == 'No tumor':
            st.balloons()
            st.sidebar.success(string)
        else:
            st.sidebar.warning(string)





