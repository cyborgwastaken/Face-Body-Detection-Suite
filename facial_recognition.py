import face_recognition
import cv2
import os
import glob
import numpy as np

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.25  # Resizing factor for frames

    def load_encoding_images(self, images_path):
        # Get list of image paths
        images_path = glob.glob(os.path.join(images_path, "*.*"))
        print("{} encoding images found.".format(len(images_path)))

        # Loop through each image and extract face encodings
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Extract file name without extension
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)

            # Encode face and add to known_face_encodings along with the name
            img_encoding = face_recognition.face_encodings(rgb_img)[0]
            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)
        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect face locations and encodings
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Compare face encodings with known faces
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # Calculate face distances to find the best match
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            # If match found, assign name
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        # Scale back face locations to original frame size
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names


# Create instance of SimpleFacerec class
sfr = SimpleFacerec()

# Load known face encodings from images folder
sfr.load_encoding_images("images/")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()

    # Detect known faces in the frame
    face_locations, face_names = sfr.detect_known_faces(frame)

    # Draw rectangles and names around detected faces
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

    # Display frame
    cv2.imshow("Frame", frame)

    # Break loop if 'Esc' key is pressed
    key = cv2.waitKey(1)
    if key == 27:
        break

# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
