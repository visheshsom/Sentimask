import torch
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2


def convert_to_corners(x_center, y_center, width, height):
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    return [x1, y1, x2, y2]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "detectionapp/model/best.pt"
model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path)
model = model.to(device)

# Gender Classification Model
gender_model_path = "detectionapp/model/gender_classification_model.h5"
gender_model = tf.keras.models.load_model(gender_model_path)


def preprocess_for_gender_model(face):
    face = face.convert("RGB")  # Convert to RGB
    face = face.resize((32, 32))  # Resize to 32x32
    face_array = np.array(face)
    face_array = face_array.astype("float32") / 255.0  # Normalize to [0, 1] range
    face_array = face_array.reshape(1, 32, 32, 3)  # Add batch dimension
    return face_array


def preprocess_for_emotion_model(face):
    face = face.convert("L")  # Convert to grayscale
    face = face.resize((48, 48))  # Resize to 48x48 for the emotion model
    face_array = np.array(face)
    face_array = face_array.reshape(1, 48, 48, 1)  # Add batch dimension
    face_array = face_array.astype("float32") / 255  # Normalize to [0, 1] range
    return face_array


# Load Emotion Model architecture from JSON
emotion_model_path = "detectionapp/model/fer.json"
with open(emotion_model_path, "r") as json_file:
    loaded_model_json = json_file.read()
emotion_model = tf.keras.models.model_from_json(loaded_model_json)

# Load weights into the emotion model
emotion_model_weights_path = "detectionapp/model/fer.h5"
emotion_model.load_weights(emotion_model_weights_path)

# Emotion labels
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def predict(image):
    results = model(image)

    # Extract prediction details
    pred_data = results.pred[0].cpu().numpy()

    # This will store our gender and emotion predictions along with positions
    predictions = []

    r_img = results.render()  # returns a list with the images as np.array
    img_with_boxes = r_img[0]  # image with boxes as np.array

    # Convert the image with bounding boxes to a format suitable for OpenCV
    img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)

    for detection in pred_data:
        x_center, y_center, width, height, _, _ = detection[:6]
        x1, y1, x2, y2 = convert_to_corners(x_center, y_center, width, height)
        cropped_face = image.crop((int(x1), int(y1), int(x2), int(y2)))

        preprocessed_face = preprocess_for_gender_model(cropped_face)

        # Predict gender
        gender_prediction = gender_model.predict(preprocessed_face)
        gender_label = "Female" if gender_prediction[0][0] < 0.5 else "Male"

        # Predict emotion
        emotion_preprocessed_face = preprocess_for_emotion_model(cropped_face)
        emotion_prediction = emotion_model.predict(emotion_preprocessed_face)
        emotion_label_idx = np.argmax(emotion_prediction[0])
        emotion_label = EMOTION_LABELS[emotion_label_idx]

        predictions.append(
            {
                "gender": gender_label,
                "emotion": emotion_label,
                "x": int(x1),
                "y": int(y1),
            }
        )

        # Display gender and emotion inside the bounding box
        gender_position = (
            int(x2 - 190),
            int(y2 + 5),
        )  # Positioned directly below the bounding box
        emotion_position = (int(x2 - 190), int(y2 + 30))
        cv2.putText(
            img_with_boxes,
            gender_label,
            gender_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img_with_boxes,
            emotion_label,
            emotion_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    # Overlay the emotion label on the top right corner
    # cv2.putText(img_with_boxes, emotion_label, (img_with_boxes.shape[1] - 150, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Convert the image back to RGB format (if you're displaying it using a tool that expects RGB format)
    img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)

    # Return the image with bounding boxes and the predictions
    return img_with_boxes, predictions
