# American Sign Language (ASL) Recognition Model

## Overview
This project focuses on recognizing American Sign Language (ASL) gestures through a user-friendly web interface. The application uses a trained machine learning model to classify hand gestures corresponding to the letters A-Z and phases individual letters into complete words.

## Features
- **Real-Time Gesture Recognition**: Utilizes webcam input to recognize and display ASL gestures in real time.
- **Interactive Interface**: Built with Streamlit for a seamless user experience.
- **Word Formation**: Accumulates recognized letters to form complete words.
- **Feedback Display**: Shows predicted letters and the final word as users perform gestures.

## Tools Used
- **Python**: Core programming language for implementation.
- **TensorFlow**: For building and deploying the machine learning model.
- **OpenCV**: For image processing and handling video input from the webcam.
- **Streamlit**: Framework for creating the web application.
- **NumPy**: For numerical computations.
- **Pandas**: For data handling (if applicable).
- **Matplotlib**: For visualizations (if applicable).

## Model Training
- **Dataset**: The model is trained on a dataset of ASL gestures for the English alphabet (A-Z).
- **Input Shape**: The model expects input in the shape of (num_samples, num_features, num_time_steps).


