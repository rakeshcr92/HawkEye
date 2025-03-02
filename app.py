from flask import Flask, render_template, request, jsonify
import cv2
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder="templates")
app.config["UPLOAD_FOLDER"] = "uploads/"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load AI model
MODEL_PATH = "fight_detection_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

def process_video(file_path, model):
    """Extracts frames from video and predicts if it's a fight or non-fight."""
    cap = cv2.VideoCapture(file_path)
    frame_sequence = []
    frame_count = 64  # Required number of frames per input

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break  # Stop if the video ends

        # Preprocess frame for the AI model
        img = cv2.resize(frame, (224, 224))  # Resize to match input shape
        img = img / 255.0  # Normalize
        frame_sequence.append(img)

        # Ensure we have 64 frames before making a prediction
        if len(frame_sequence) == frame_count:
            # Convert to NumPy array with correct shape (1, 64, 224, 224, 3)
            input_data = np.expand_dims(np.array(frame_sequence), axis=0)

            # Make prediction
            prediction = model.predict(input_data)
            predicted_class = np.argmax(prediction)

            # Clear frame buffer for next sequence
            frame_sequence = []

            if predicted_class == 1:
                cap.release()
                return "Fight detected!"

    cap.release()
    return "No fight detected."

@app.route('/incidents')
def incidents():
    return render_template("incidents.html")

@app.route('/')
def home():
    return render_template("main.html")

@app.route('/predict', methods=["POST"])
def predict():
    """Handle video upload and return AI prediction."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # Process the uploaded video and pass the model
    prediction = process_video(file_path, model)

    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(debug=True)

