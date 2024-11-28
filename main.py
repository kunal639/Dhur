import cv2
from deepface import DeepFace
import csv
from collections import Counter
from itertools import count

# Mapping emotions to numeric values
emotion_to_number = {
    'angry': 1.0,
    'disgust': 2.0,
    'fear': 3.0,
    'happy': 4.0,
    'sad': 5.0,
    'surprise': 6.0,
    'neutral': 7.0
}

# Initialize face tracker and ID counter
face_tracker = {}
id_counter = count(1)

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video
cap = cv2.VideoCapture(1)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale for detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    current_faces = []

    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = rgb_frame[y:y + h, x:x + w]

        # Perform emotion analysis
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        numeric_value = emotion_to_number.get(emotion, 0.0)

        # Identify or assign ID to the face
        face_center = (x + w // 2, y + h // 2)
        face_id = None

        # Match current face with tracked faces
        for tracked_id, tracked_center in face_tracker.items():
            if abs(tracked_center[0] - face_center[0]) < 50 and abs(tracked_center[1] - face_center[1]) < 50:
                face_id = tracked_id
                break

        # If no match found, assign a new ID
        if face_id is None:
            face_id = next(id_counter)

        # Update tracker
        face_tracker[face_id] = face_center

        # Save emotion to respective CSV
        student_file = f'student_{face_id}.csv'
        with open(student_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([emotion, numeric_value])

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {face_id} {emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Real-time Multi-Face Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Summarize emotions for each student
for tracked_id in face_tracker.keys():
    student_file = f'student_{tracked_id}.csv'
    summary_file = f'student_{tracked_id}_summary.csv'
    emotion_counts = Counter()

    with open(student_file, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            emotion = row[0]
            emotion_counts[emotion] += 1

    with open(summary_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Emotion', 'Count'])
        for emotion, count in emotion_counts.items():
            writer.writerow([emotion, count])

print("Emotion summaries created for all students.")
