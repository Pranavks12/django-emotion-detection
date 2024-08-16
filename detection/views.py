# detection/views.py

from django.shortcuts import render
from django.http import StreamingHttpResponse
import cv2
from fer import FER
import tensorflow as tf

# Set TensorFlow to use CPU only
tf.config.set_visible_devices([], 'GPU')

# Initialize the FER detector with MTCNN for better accuracy
detector = FER(mtcnn=True)

def generate_frames():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect emotions in the frame
        result = detector.detect_emotions(frame)

        # Draw bounding boxes and emotion labels on the frame
        for face in result:
            (x, y, w, h) = face['box']
            emotions = face['emotions']
            top_emotion = max(emotions, key=emotions.get)
            score = emotions[top_emotion]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{top_emotion}: {score:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Encode the frame in JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame to the browser
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

def video_feed(request):
    return StreamingHttpResponse(generate_frames(),
                                 content_type='multipart/x-mixed-replace; boundary=frame')

def index(request):
    return render(request, 'detection/index.html')
