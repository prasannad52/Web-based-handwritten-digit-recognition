from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import cv2
import base64

# Initialize Flask app
app = Flask(__name__)

# Load your pre-trained model
model = tf.keras.models.load_model("mnist_cnn_model.h5")

def preprocess_image(image_data):
    """
    Preprocess the base64 image data to prepare it for model prediction.
    """
    # Decode base64 image
    image_decoded = base64.b64decode(image_data.split(",")[1])
    
    # Convert to NumPy array
    image_array = np.frombuffer(image_decoded, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
    
    # Resize to 28x28
    image_resized = cv2.resize(image, (28, 28))
    
    # Normalize pixel values to [0, 1]
    image_normalized = image_resized / 255.0
    
    # Expand dimensions to match model input shape
    image_final = np.expand_dims(image_normalized, axis=(0, -1))
    
    return image_final

@app.route("/")
def home():
    """
    Render the HTML template with the canvas interface.
    """
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict the digit from the drawn image.
    """
    try:
        # Get the image data from the request
        data = request.get_json()
        image_data = data["image"]
        
        # Preprocess the image
        preprocessed_image = preprocess_image(image_data)
        
        # Predict the digit
        predictions = model.predict(preprocessed_image)
        predicted_digit = int(np.argmax(predictions))
        
        return jsonify({"digit": predicted_digit})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
