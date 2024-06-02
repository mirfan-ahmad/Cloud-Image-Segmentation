import tensorflow as tf
from unet import Unet

import numpy as np
import matplotlib.pyplot as plt

import os
import cv2
import streamlit as st


def Display(original_image, predictions):
    class_names = {0: 'CLEAR', 1: 'CLOUD', 2: 'CLOUD_SHADOW'}
    fig, axes = plt.subplots(1, predictions.shape[-1] + 1, figsize=(15, 5))
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    for i in range(predictions.shape[-1]):
        binary_mask = np.where(predictions[:, :, i] > 0.5, 1, 0)
        axes[i+1].imshow(binary_mask, cmap='gray')
        axes[i+1].set_title(f'{class_names[i]}')
        axes[i+1].axis('off')

    plt.tight_layout()
    return fig

def Predict(file):
    npy = np.load(file)
    img = cv2.resize(npy[..., [2, 3, 4]], (256, 256), interpolation = cv2.INTER_NEAREST)
    img = np.expand_dims(img, axis=0)
    
    with st.spinner('Segmenting...'):
        # Initiate the model
        model = Unet()
        model = model.build_unet_model()
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss="categorical_crossentropy",
                    metrics=["accuracy"])
        
        # Load the weights
        model.load_weights("unet_trained.h5")
        
        # Predict the mask
        prediction = model.predict(img)
    
        # Display all the channels of mask
        return Display(img.squeeze(), prediction.squeeze())


if __name__ == '__main__':
    st.set_page_config(page_title="Cloud Segmentation", page_icon=":cloud:", layout="wide")
    st.title("Cloud Segmentation - Mask Generator")
    file = st.file_uploader("Upload an image", type=["npy"])
    
    if st.button('Segment') and file:
        fig = Predict(file)
        st.pyplot(fig)
    
    if st.button('Segment on sample image'):
        image_path = os.listdir('sample images')[np.random.randint(0, 5)]
        sample_file = os.path.join('sample images', image_path)

        if len(sample_file) > 0:
            fig = Predict(sample_file)
            st.pyplot(fig)