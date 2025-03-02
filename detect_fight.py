import cv2
import tensorflow as tf
import numpy as np

# Load the trained AI model
MODEL_PATH = "fight_detection_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Initialize webcam
cap = cv2.VideoCapture(1)  # 0 for default webcam

frame_sequence = []
frame_count = 64  # Number of frames required for model input

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if no frame is captured

    # Preprocess frame for the model
    img = cv2.resize(frame, (224, 224))  # Resize to match model input size
    img = img / 255.0  # Normalize
    frame_sequence.append(img)

    # Keep only the last 64 frames
    if len(frame_sequence) > frame_count:
        frame_sequence.pop(0)

    # If we have enough frames, make a prediction
    if len(frame_sequence) == frame_count:
        input_data = np.expand_dims(np.array(frame_sequence), axis=0)  # Shape: (1, 64, 224, 224, 3)
        prediction = model.predict(input_data)
        predicted_class = np.argmax(prediction)  # 0 = No Violence, 1 = Fight Detected

        # Display prediction on the frame
        label = "Anamoly Detected!" if predicted_class == 1 else "No Violence"
        color = (0, 0, 255) if predicted_class == 1 else (0, 255, 0)

        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    # Show the live feed
    cv2.imshow("Live Fight Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
