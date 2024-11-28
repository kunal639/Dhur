import cv2
from deepface import DeepFace
import csv
from collections import Counter

# Mapping emotions to numeric values (can be integer or floating point)
emotion_to_number = {
    'angry': 1.0,
    'disgust': 2.0,
    'fear': 3.0,
    'happy': 4.0,
    'sad': 5.0,
    'surprise': 6.0,
    'neutral': 7.0
}

# Create/open a CSV file to log emotions
emotion_log_file = 'emotion_log.csv'
with open(emotion_log_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Frame', 'Emotion', 'Numeric Value'])

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video
cap = cv2.VideoCapture(0)
frame_count = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale frame to RGB format
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = rgb_frame[y:y + h, x:x + w]

        # Perform emotion analysis on the face ROI
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

        # Determine the dominant emotion
        emotion = result[0]['dominant_emotion']
        numeric_value = emotion_to_number.get(emotion, 0.0)  # Get numeric value, default to 0 if not found

        # Write the dominant emotion and its numeric value to the file
        with open(emotion_log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([frame_count, emotion, numeric_value])

        # Draw rectangle around face and label with predicted emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Increment the frame count
    frame_count += 1

    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()

# Summarizing emotions from the log file
emotion_counts = Counter()

with open(emotion_log_file, mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        emotion = row[1]
        emotion_counts[emotion] += 1

# Write the summary to another CSV file
summary_file = 'emotion_summary.csv'
with open(summary_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Emotion', 'Count'])
    for emotion, count in emotion_counts.items():
        writer.writerow([emotion, count])

print("Emotion summary saved to:", summary_file)
