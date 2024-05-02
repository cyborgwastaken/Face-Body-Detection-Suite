# Face-Body-Detection-Suite - AI & Machine Learning Project

Welcome to the AI & Machine Learning project developed by Group of 1st Year students from VIT Bhopal University. This project comprises three sub-projects focusing on different aspects of computer vision and artificial intelligence:

1. **Facial Recognition** 👤
2. **Body Movement & Posture Detection** 🏃‍♂️
3. **Face Extractor** 🖼️

Each sub-project is described below along with its implementation details.

---

## Facial Recognition

### Description:
Facial recognition is the process of identifying or verifying the identity of a person using their face. In this project, we utilize the face_recognition library along with OpenCV to recognize faces in real-time using a webcam.

### Implementation:
- The `SimpleFacerec` class is implemented to handle facial recognition tasks.
- It loads known face encodings from images provided in a specified folder.
- It then detects known faces in real-time video streams and annotates them with names.

### Usage:
- Ensure Python and required libraries are installed.
- Place images of known faces in the `images/` folder.
- Run the `facial_recognition.py` script.

---

## Body Movement & Posture Detection

### Description:
Body movement and posture detection aim to recognize and analyze human body gestures and postures. In this project, we use the Mediapipe library to detect key body landmarks and hand movements in real-time.

### Implementation:
- We utilize the `Holistic` model from the Mediapipe library to detect body landmarks and hand gestures.
- Detected landmarks are annotated on the video stream in real-time.

### Usage:
- Ensure Python and required libraries are installed.
- Run the `body_movement_detection.py` script.

---

## Face Extractor

### Description:
The face extractor sub-project focuses on detecting and extracting faces from images. We utilize the Haar cascade classifier for face detection and crop the detected faces from the original image.

### Implementation:
- Haar cascade classifier is employed for face detection in images.
- Detected faces are cropped and saved as separate images.

### Usage:
- Ensure Python and required libraries are installed.
- Place the image file to be processed in the project directory.
- Run the `face_extractor.py` script.

---

## Requirements:
- Python 3.11
- OpenCV
- face_recognition
- Mediapipe
- Haarcascade

---

Feel free to explore each sub-project's directory for detailed implementation and additional notes.

For any questions or feedback, please contact the project contributors.

Enjoy exploring AI & Machine Learning with us!

