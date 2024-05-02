import cv2
import os

# Function to create a folder if it doesn't exist
def create_folder(folder_name):
    try:
        # Get the current working directory
        root_directory = os.getcwd()

        # Create a new directory path
        new_directory = os.path.join(root_directory, folder_name)

        # Check if the directory already exists, if not, create it
        if not os.path.exists(new_directory):
            os.makedirs(new_directory)
            print(f"Folder '{folder_name}' created successfully.")
        else:
            print(f"Folder '{folder_name}' already exists.")
    except Exception as e:
        print(f"Error occurred: {e}")

# Path to the Haar cascade file for face detection
alg = "haarcascades/haarcascade_frontalface_default.xml"

# Create Haar cascade classifier object
haar_cascade = cv2.CascadeClassifier(alg)

# File name of the image to process
file_name = "test.png"

# Read the image in grayscale mode
img = cv2.imread(file_name, 0)

# Convert the grayscale image to BGR (OpenCV's default color format)
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# Detect faces in the image
faces = haar_cascade.detectMultiScale(
    gray_img, scaleFactor=1.05, minNeighbors=2, minSize=(100, 100)
)

# Create a folder to store the cropped faces
folder_name = os.path.splitext(file_name)[0]
create_folder(f"stored-faces/{folder_name}")

# Counter for naming the cropped face images
i = 0

# Loop through each detected face
for x, y, w, h in faces:
    # Crop the face from the original image
    cropped_image = img[y: y + h, x: x + w]

    # Define the target file name for the cropped face
    target_file_name = f"stored-faces/{folder_name}/{i}.jpg"

    # Save the cropped face as a separate image
    cv2.imwrite(target_file_name, cropped_image)

    # Increment counter for the next face
    i += 1
