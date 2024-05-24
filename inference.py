import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
from hparams import *
from model import get_model
import time

model = get_model("lips_model_ep_2.h5")

# Open the camera
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")
# used to record the time when we processed last frame 
prev_frame_time = 0
  
# used to record the time at which we processed current frame 
new_frame_time = 0
while (True): 
    ret, frame = cap.read()
    # Convert the image to PIL format
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = image.resize(IMAGE_SIZE)

    image = np.array(image)
    image = tf.expand_dims(image, axis=0)
    predict_masks = model.predict(image)

    # Apply the mask to the original frame
    mask = cv2.resize(predict_masks[0], (frame.shape[1], frame.shape[0]))
    mask = np.uint8(255 * mask)
    mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    
    
    # time when we finish processing for this frame 
    new_frame_time = time.time() 
    fps = 1/(new_frame_time-prev_frame_time) 
    prev_frame_time = new_frame_time 
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    frame_with_mask = cv2.addWeighted(frame, 0.7, mask_color, 0.3, 0)
        # font which we will be using to display FPS 

    # Display the frame with mask
    cv2.imshow('Frame with Mask', frame_with_mask)

    # Break the loop if 'q' is pressed or the window is closed
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Frame with Mask', cv2.WND_PROP_VISIBLE) < 1:
        break
# Release the camera
cap.release()
cv2.destroyAllWindows()