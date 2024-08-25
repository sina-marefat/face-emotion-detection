import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
image_size = (48, 48)
model = load_model('emotion_model.h5')

# Load class labels
class_labels = ['happy', 'sad', 'angry', 'neutral','surprise']

def preprocess_image(image):
    image = cv2.resize(image, image_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_filename = f'captured_image.png'
    cv2.imwrite(image_filename, image)
    image = image.astype('float32') / 255
    image = np.expand_dims(image, axis=0)
    return image

def predict_emotion(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    return class_labels[np.argmax(prediction)]

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    emotion = predict_emotion(frame)
    cv2.putText(frame, emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()