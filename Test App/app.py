import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

# Load models
vehicle_tracking_model = load_model('vehicle_tracking_model.h5', compile=False)
traffic_light_control_model = load_model('traffic_light_control_model.h5', compile=False)
congestion_detection_model = load_model('congestion_detection_model.h5', compile=False)

# Model options
model_options = {
    "Vehicle Tracking": vehicle_tracking_model,
    "Traffic Light Control": traffic_light_control_model,
    "Congestion Detection": congestion_detection_model
}

st.title("Traffic Management System")

# Select video input method
video_input_option = st.radio("Select Video Input Method", ("Live Stream", "Upload Video"))

# Video upload
uploaded_video = None
if video_input_option == "Upload Video":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi"])

# Select models
selected_models = st.multiselect("Select Models", list(model_options.keys()))

# Function to overlay text on video frames at the bottom left
def overlay_text(frame, texts, font_scale=0.6, color=(0, 0, 0), thickness=1):
    y0, dy = 30, 30  # Initial position and line height
    for i, text in enumerate(texts):
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        position = (10, y0 + i * dy)
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

# Function to process video and overlay predictions in real-time
def process_video(cap, models, selected_models):
    frame_window = st.image([])  # Initialize an empty frame
    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("No more frames available or camera disconnected.")
            break

        # Convert frame to grayscale if necessary
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize frame to match model's expected input shape (4, 1)
        resized_frame = cv2.resize(gray_frame, (4, 1))
        resized_frame = np.expand_dims(resized_frame, axis=-1)  # Add channel dimension (4, 1, 1)
        resized_frame = np.expand_dims(resized_frame, axis=0)  # Add batch dimension (1, 4, 1, 1)
        resized_frame = np.squeeze(resized_frame, axis=1)  # Adjust to (1, 4, 1)

        # Collect texts for overlay
        texts = []
        for model_name in selected_models:  # Only iterate over selected models
            model = models[model_name]
            prediction = model.predict(resized_frame)[0][0]  # Get the prediction value

            # Determine the text to display based on the prediction
            if model_name == "Vehicle Tracking":
                text = "Vehicle Detected" if prediction >= 0.5 else "No Vehicle Detected"
                texts.append(text)
            
            elif model_name == "Traffic Light Control":
                text = "Green Light" if prediction >= 0.5 else "Red Light"
                texts.append(text)
            
            elif model_name == "Congestion Detection":
                text = "Congestion Detected" if prediction >= 0.5 else "No Congestion Detected"
                texts.append(text)

        # Display all collected texts on the frame
        overlay_text(frame, texts)

        # Display the frame with overlay in Streamlit
        frame_window.image(frame, channels="BGR")

        # Add a small delay to simulate real-time processing
        time.sleep(0.03)

    cap.release()

# Process live stream or uploaded video
if video_input_option == "Live Stream":
    st.write("Starting live stream...")
    cap = cv2.VideoCapture(0)  # 0 is typically the default camera
    if not cap.isOpened():
        st.write("Failed to open camera.")
    else:
        process_video(cap, model_options, selected_models)

elif video_input_option == "Upload Video" and uploaded_video is not None:
    st.write(f"Processing video: {uploaded_video.name}")
    video_path = uploaded_video.name
    # Save uploaded file temporarily
    with open(video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())
    # Process video
    process_video(cv2.VideoCapture(video_path), model_options, selected_models)
    # Optionally, remove the temporary file after processing
