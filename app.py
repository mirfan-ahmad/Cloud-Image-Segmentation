import streamlit as st
import numpy as np
import tarfile
import os
import cv2
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt

from model import UNet


MODEL_PATH = 'unet_model.h5'
SEGMENTED_OUTPUT_PATH = "segmented_output"


@st.cache(allow_output_mutation=True)
def load_unet_model():
    model = load_model(MODEL_PATH)
    return model

def validate_file(uploaded_file):
    if uploaded_file.name.endswith('.npy') or uploaded_file.name.endswith('.tar'):
        return True
    return False

def process_file(uploaded_file):
    if uploaded_file.name.endswith('.npy'):
        image = np.load(uploaded_file)
        return image
    elif uploaded_file.name.endswith('.tar'):
        with tarfile.open(fileobj=uploaded_file) as tar:
            tar.extractall()
            image_files = [os.path.join(SEGMENTED_OUTPUT_PATH, f) for f in os.listdir(SEGMENTED_OUTPUT_PATH) if f.endswith('.npy')]
            images = [np.load(image_file) for image_file in image_files]
            return np.stack(images, axis=0)
    else:
        st.error("Unsupported file format!")
        return None

def segment_image(image, model):
    image_resized = cv2.resize(image, (256, 256), interpolation=cv2.INTER_NEAREST)
    image_batch = np.expand_dims(image_resized, axis=0)
    segmented = model.predict(image_batch)
    return segmented[0]

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
    st.pyplot(fig)


def main():
    st.title("Image Segmentation with U-Net")
    uploaded_file = st.file_uploader("Upload a .npy or .tar file", type=['npy', 'tar'])
    
    if uploaded_file is not None:
        if validate_file(uploaded_file):
            st.success("File validated successfully.")
            image = process_file(uploaded_file)

            if image is not None:
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                if st.button("Segment Image"):
                    model = load_unet_model()
                    segmented_image = segment_image(image, model)
                    
                    st.image(segmented_image, caption="Segmented Image", use_column_width=True)
                    if st.button("Display Segmented Channels"):
                        Display(image, segmented_image)
        else:
            st.error("Invalid file. Please upload a .npy or .tar file.")

if __name__ == "__main__":
    main()
