"""Smile Detection System - Streamlit app (OpenCV Haar Cascade only)."""

import cv2
import numpy as np
import streamlit as st
from PIL import Image


# ---------------------------------------------------------------
# Core detection logic (identical to the notebook)
# ---------------------------------------------------------------
def load_cascades():
    """Load the pretrained Haar Cascade classifiers shipped with OpenCV."""
    face_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    smile_path = cv2.data.haarcascades + "haarcascade_smile.xml"

    face = cv2.CascadeClassifier(face_path)
    smile = cv2.CascadeClassifier(smile_path)

    if face.empty():
        raise RuntimeError("Face Haar Cascade could not be loaded.")
    if smile.empty():
        raise RuntimeError("Smile Haar Cascade could not be loaded.")
    return face, smile


def detect_faces_and_smiles(frame, face_cascade, smile_cascade):
    """Detect faces, then smiles inside each face.

    Returns (annotated_frame, number_of_faces, number_of_smiles).
    `frame` must be a BGR image (the OpenCV default).
    """
    output = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 1: find faces in the whole image
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60)
    )

    total_smiles = 0
    for (x, y, w, h) in faces:
        # Draw the face box and label
        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(output, "Face", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Step 2: search for smiles ONLY inside the lower half of the face
        # (smiles are never in the forehead/eyes; this reduces false positives)
        roi_top = y + h // 2
        face_gray = gray[roi_top:y + h, x:x + w]

        smiles = smile_cascade.detectMultiScale(
            face_gray, scaleFactor=1.7, minNeighbors=20, minSize=(25, 25)
        )

        for (sx, sy, sw, sh) in smiles:
            # Convert ROI coordinates back to full-frame coordinates
            fx, fy = x + sx, roi_top + sy
            cv2.rectangle(output, (fx, fy), (fx + sw, fy + sh), (0, 255, 0), 2)
            cv2.putText(output, "Smile", (fx, fy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            total_smiles += 1

    return output, len(faces), total_smiles


# ---------------------------------------------------------------
# Streamlit helpers
# ---------------------------------------------------------------
@st.cache_resource
def get_cascades():
    """Load cascades once and reuse them between reruns."""
    return load_cascades()


def read_image(file):
    """Convert an uploaded/captured file into a BGR NumPy array."""
    image = Image.open(file).convert("RGB")      # PIL gives RGB
    rgb = np.array(image)
    if rgb.size == 0:
        raise ValueError("Empty image.")
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # OpenCV needs BGR


def process_image(file, face_cascade, smile_cascade):
    """Read the file, run detection and show the results."""
    try:
        bgr = read_image(file)
    except Exception:
        st.error("Invalid or unsupported image. Please upload a valid JPG or PNG.")
        return

    result_bgr, n_faces, n_smiles = detect_faces_and_smiles(
        bgr, face_cascade, smile_cascade
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col2:
        st.subheader("Processed Image")
        st.image(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.subheader("Detection Results")
    m1, m2 = st.columns(2)
    m1.metric("Faces Detected", n_faces)
    m2.metric("Smiles Detected", n_smiles)

    if n_faces == 0:
        st.warning("No face detected in the image.")
    elif n_smiles == 0:
        st.info("No smile detected.")
    else:
        st.success("Smile detected!")


# ---------------------------------------------------------------
# Main app
# ---------------------------------------------------------------
def main():
    st.set_page_config(page_title="Smile Detection System", page_icon="😊")
    st.title("😊 Smile Detection System")
    st.write("Detect faces and smiles using OpenCV Haar Cascade")

    try:
        face_cascade, smile_cascade = get_cascades()
    except RuntimeError as err:
        st.error(str(err))
        st.stop()

    mode = st.radio("Select Input Mode", ["Image Upload", "Webcam"], horizontal=True)

    if mode == "Image Upload":
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            process_image(uploaded, face_cascade, smile_cascade)
        else:
            st.info("Please upload a JPG, JPEG or PNG image.")
    else:
        st.caption("Note: this captures a single photo from your webcam; "
                   "it is not continuous video streaming.")
        photo = st.camera_input("Take a picture")
        if photo is not None:
            process_image(photo, face_cascade, smile_cascade)
        else:
            st.info("Allow camera access in your browser, then click 'Take photo'.")

    st.markdown("---")
    st.caption("Haar Cascade is a classical technique; results depend on lighting, "
               "face angle, image quality and occlusion.")


if __name__ == "__main__":
    main()
