import face_recognition
import cv2
import threading

# Load the known image and encode the face
known_image_path = r'D:\python\face_recognition\images\yahya img.jpg'
known_image = face_recognition.load_image_file(known_image_path)
known_face_encoding = face_recognition.face_encodings(known_image)[0]

# Initialize variables
face_locations = []
face_encodings = []
face_names = []

# Function for face recognition
def recognize_faces(frame):
    global face_locations, face_encodings, face_names

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces([known_face_encoding], face_encoding)
        name = "Unknown"
        if True in matches:
            name = "Known"
        face_names.append(name)

# Function for video capture
def capture_video():
    video_capture = cv2.VideoCapture(0)

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize the frame to speed up face detection (optional)
        # frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Recognize faces
        recognize_faces(frame)

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

# Run video capture and face recognition in separate threads
video_thread = threading.Thread(target=capture_video)
video_thread.start()
