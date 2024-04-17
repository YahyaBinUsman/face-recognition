import cv2
import numpy as np

# Load pre-trained face detection and recognition models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the known face image and its corresponding label
known_face_image = cv2.imread('python/yahya img.jpg')

if known_face_image is None:
    print("Error: Unable to load the image.")
else:
    print("Image loaded successfully.")

known_face_label = 0  # Assign a label to the known face

# Convert the known face image to grayscale
gray_known_face = cv2.cvtColor(known_face_image, cv2.COLOR_BGR2GRAY)

# Train the recognizer with the known face image and label
recognizer.train([gray_known_face], np.array([known_face_label]))

# Open a video capture device (webcam)
video_capture = cv2.VideoCapture(0)  # Use 0 for default webcam, or replace with video file path

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face_roi = gray_frame[y:y+h, x:x+w]

        # Recognize the face using the trained recognizer
        label, confidence = recognizer.predict(face_roi)

        # Determine if the face is known or unknown based on confidence threshold
        if confidence < 100:  # Adjust this threshold as needed
            # Face is known
            cv2.putText(frame, "Known", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            # Face is unknown
            cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the frame with detected faces
    cv2.imshow('Face Recognition', frame)

    # Check for exit key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and close all windows
video_capture.release()
cv2.destroyAllWindows()
