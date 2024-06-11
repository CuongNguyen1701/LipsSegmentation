import streamlit as st
from PIL import Image
from hparams import *
from model import get_model
import cv2
import time
import numpy as np
import tensorflow as tf
from cmap import create_colormap

if "model" not in st.session_state.keys():
    st.session_state["model"] = get_model("lips_model_unet.h5")

model = st.session_state["model"]

st.title("Webcam Live Feed")

run = st.toggle("Run", False)
color = st.color_picker("Pick A Color", "#FF0000")
FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

colormap = create_colormap("#000000", color)

# Initialize variables for FPS calculation
start_time = time.time()
frame_count = 0

fps_text = st.text("FPS: ")

while run:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    image = Image.fromarray(rgb_frame)
    image = image.resize(IMAGE_SIZE)
    image = np.array(image)
    image = tf.expand_dims(image, axis=0)
    predict_masks = model.predict(image)

    # Apply the mask to the original frame
    mask = cv2.resize(predict_masks[0], (frame.shape[1], frame.shape[0]))
    mask = np.uint8(255 * mask)
    
    mask_color = cv2.applyColorMap(mask, colormap)
    
    frame_with_mask = cv2.addWeighted(src1=frame, alpha=0.7, src2=mask_color, beta=0.3, gamma=0)

    rgb_frame_with_mask = cv2.cvtColor(frame_with_mask, cv2.COLOR_BGR2RGB)
    
    # Calculate FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    
    # Update the FPS text element
    fps_text.text(f"FPS: {fps:.2f}")
    
    FRAME_WINDOW.image(rgb_frame_with_mask)
    
    # Reset start time and frame count
    if elapsed_time >= 1.0:
        start_time = time.time()
        frame_count = 0
else:
    st.write('Stopped')
