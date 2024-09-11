# Cloud Image Segmentation with U-Net

This repository contains a **Streamlit** application that allows users to upload `.npy` or `.tar` files containing satellite images, segment the images using a pre-trained **U-Net** model, and visualize the segmentation results. The U-Net architecture is specifically tailored for multi-class image segmentation, identifying classes like `CLEAR`, `CLOUD`, and `CLOUD_SHADOW` in satellite imagery.

## Features

- Upload **.npy** or **.tar** files containing satellite images.
- Use a pre-trained **U-Net** model to segment the uploaded images.
- Display the original image alongside the segmented masks for each class (`CLEAR`, `CLOUD`, `CLOUD_SHADOW`).
- View individual channels of the segmented output for a better understanding of the model’s predictions.

## Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.7+
- Required Python packages (see below)

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/mirfan-ahmad/Cloud-Image-Segmentation.git
   cd Cloud-Image-Segmentation
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the pre-trained U-Net model (`unet_model.h5`) and place it in the root directory of the project.

### Run the Application

1. Launch the **Streamlit** app:

   ```bash
   streamlit run app.py
   ```

2. In the web browser, you can now upload `.npy` or `.tar` files, perform segmentation, and view the segmented results.

### Folder Structure

```bash
├── app.py                     # Streamlit application code
├── model.py                   # U-Net model class definition
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── unet_model.h5              # Pre-trained U-Net model (download separately)
```

### Example Input/Output

- **Input**: Satellite image in `.npy` or `.tar` format.
- **Output**: Segmented mask showing `CLEAR`, `CLOUD`, and `CLOUD_SHADOW` regions.

### How It Works

1. **File Upload**: Users can upload `.npy` or `.tar` files containing satellite images. The app validates the file format and loads the image(s) for processing.
  
2. **Segmentation**: The app uses the pre-trained U-Net model to perform segmentation on the uploaded image. The model predicts pixel-wise class labels for `CLEAR`, `CLOUD`, and `CLOUD_SHADOW`.
  
3. **Visualization**: The original image and the segmented output are displayed side by side. The user can also view each segmented class as an individual binary mask.

### Example Workflow

1. Upload your `.npy` or `.tar` file containing satellite images.
2. Click the "Segment Image" button to apply the U-Net model for segmentation.
3. View the segmented result.
4. Click the "Display Segmented Channels" button to view each class (CLEAR, CLOUD, CLOUD_SHADOW) as a binary mask.

## Dependencies

To install all the required Python libraries, run:

```bash
pip install -r requirements.txt
```

The main dependencies are:

- `streamlit`: To create the interactive app interface.
- `tensorflow`: To load and use the pre-trained U-Net model.
- `matplotlib`: To visualize the results.
- `opencv-python`: To handle image processing tasks.

## U-Net Model

The U-Net model is built for image segmentation, designed to handle multi-class image segmentation tasks. The architecture consists of an encoder (downsampling path) and decoder (upsampling path), with skip connections that help retain important spatial features.

You can find the architecture details in the `model.py` file.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
