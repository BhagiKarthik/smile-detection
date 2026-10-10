# Smile Detection System Using Haar Cascade

## Description
A beginner-friendly computer vision project that detects faces and then smiles inside each face, using OpenCV's pretrained Haar Cascade classifiers (`haarcascade_frontalface_default.xml` and `haarcascade_smile.xml`). No model training is involved. Includes a Jupyter Notebook and a Streamlit web app.

## Technologies
* Python
* OpenCV
* Haar Cascade
* NumPy
* Streamlit

## Installation
```bash
pip install -r requirements.txt
```

## Run Notebook
Open `Smile_Detection_System.ipynb` in Jupyter Notebook or VS Code and run the cells from top to bottom. Put a test image named `sample.jpg` in the same folder. The webcam cell needs a local webcam; press `q` to quit.

## Run Streamlit
```bash
streamlit run app.py
```

## Features
* Face detection
* Smile detection
* Image input (JPG, JPEG, PNG)
* Webcam capture (single photo via `st.camera_input`, not continuous video)
* Bounding boxes and labels
* Detection counts
* Streamlit interface

## Limitations
Haar Cascade performance can be affected by:
* Lighting
* Face orientation
* Image resolution
* Occlusion
* Camera quality
* False positives (and missed smiles)
