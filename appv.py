import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os

#  pre-trained model
model = load_model('asl_sign_language_model_v2.h5')

# dictionary for letter predictions
letters = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
           10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 
           19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

# Function to predict letter from image
def predict_letter(image):
    img = cv2.resize(image, (150, 150))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Predict the letter
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)
    
    return letters[predicted_class[0]]

# Function to extract frames from video
def extract_frames_from_video(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames

# Function to predict letters from video frames
def predict_from_video(video_frames):
    predicted_phrase = []
    for frame in video_frames:
        predicted_letter = predict_letter(frame)
        predicted_phrase.append(predicted_letter)
    return ''.join(predicted_phrase)

# Function to predict letters from multiple uploaded images
def predict_from_images(images):
    predicted_phrase = []
    for image in images:
        image_array = np.array(image)
        predicted_letter = predict_letter(image_array)
        predicted_phrase.append(predicted_letter)
    return ''.join(predicted_phrase)

# UI code
st.title("Sign Language Recognition")

# Option to upload an image or a video or multiple images
input_type = st.radio("Choose input type", ("Single Image", "Multiple Images", "Video"))

if input_type == "Single Image":
    uploaded_file = st.file_uploader("Upload a sign language image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        
        predicted_letter = predict_letter(image)
        st.image(image, caption=f"Predicted Sign Language Letter: {predicted_letter}", use_column_width=True)
        
elif input_type == "Multiple Images":
    uploaded_files = st.file_uploader("Upload multiple sign language images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
    
    if uploaded_files:
        images = []
        
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            images.append(image)
        
        predicted_phrase = predict_from_images(images)
        
        st.write("Uploaded Images and Predictions:")
        for i, img in enumerate(images):
            st.image(img, caption=f"Predicted Letter: {predicted_phrase[i]}", use_column_width=True)
        
        st.write(f"Predicted Sign Language Phrase: {predicted_phrase}")
        
elif input_type == "Video":
    uploaded_video = st.file_uploader("Upload a sign language video", type=['mp4', 'mov', 'avi'])
    
    if uploaded_video is not None:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.getbuffer())
        
        frames = extract_frames_from_video("temp_video.mp4")
        
        predicted_phrase = predict_from_video(frames)
        
        st.video(uploaded_video)
        st.write(f"Predicted Sign Language Phrase: {predicted_phrase}")

if os.path.exists("temp_video.mp4"):
    os.remove("temp_video.mp4")
