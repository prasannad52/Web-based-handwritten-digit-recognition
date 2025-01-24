import os
import base64
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for, flash, session
from tensorflow.keras.models import load_model
from pymongo import MongoClient
import bcrypt
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # For session management and flash messages

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["HDR"]
users_collection = db["users"]

# Register route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if not username or not password:
            flash("Username and password are required!", "error")
            return redirect(url_for('register'))

        if users_collection.find_one({"username": username}):
            flash("Username already exists!", "error")
            return redirect(url_for('register'))

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        users_collection.insert_one({"username": username, "password": hashed_password})
        flash("User registered successfully!", "success")
        return redirect(url_for('login'))

    return render_template('register.html')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = users_collection.find_one({"username": username})
        if not user or not bcrypt.checkpw(password.encode('utf-8'), user["password"]):
            flash("Invalid username or password!", "error")
            return redirect(url_for('login'))

        session['username'] = username
        flash("Login successful!", "success")
        return redirect(url_for('dashboard'))

    return render_template('login.html')

# Dashboard route
@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        flash("You must log in to access the dashboard!", "error")
        return redirect(url_for('login'))

    return render_template('dashboard.html', username=session['username'])

# Logout route
@app.route('/logout')
def logout():
    session.pop('username', None)
    flash("You have been logged out!", "success")
    return redirect(url_for('login'))

# Load the pre-trained model
model = load_model('mnist_cnn_model.h5')

def preprocess_canvas_image(data_url):
    """
    Preprocess the canvas image:
    1. Decode base64 image.
    2. Convert to grayscale and threshold for white-on-black.
    3. Segment digits and resize to 28x28.
    """
    content = data_url.split(',')[1]
    image_data = base64.b64decode(content)
    np_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)

    _, image = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digit_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > 20:  # Ignore small noise
            digit = image[y:y+h, x:x+w]
            digit = cv2.resize(digit, (28, 28))
            digit = digit / 255.0  # Normalize to range [0, 1]
            digit_images.append((x, digit))

    digit_images = sorted(digit_images, key=lambda x: x[0])
    return [img[1] for img in digit_images]

def preprocess_webcam_image(frame):
    """
    Preprocess the webcam frame:
    1. Convert to grayscale and threshold for white-on-black.
    2. Find contours and resize each digit to 28x28.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digit_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > 20 and w > 20:  # Ignore small noise
            digit = thresh[y:y+h, x:x+w]
            digit = cv2.resize(digit, (28, 28))
            digit = digit / 255.0  # Normalize to range [0, 1]
            digit_images.append((x, digit))

    digit_images = sorted(digit_images, key=lambda x: x[0])
    return [img[1] for img in digit_images]

def preprocess_uploaded_image_for_multiple_digits(image):
    """
    Preprocess an uploaded image containing multiple digits:
    1. Convert the image to grayscale.
    2. Apply thresholding to get a binary image.
    3. Find contours for digit segmentation.
    4. Resize each detected digit to 28x28 pixels.
    5. Normalize pixel values to the range [0, 1].
    6. Return all digit images as a list.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to create a binary image
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours to detect each digit in the image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digit_images = []

    for contour in contours:
        # Get the bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)
        
        if h > 15 and w > 15:  # Filter out noise based on size
            # Extract the digit from the image
            digit = thresh[y:y + h, x:x + w]
            
            # Resize the digit to 28x28
            digit_resized = cv2.resize(digit, (28, 28))
            
            # Normalize the digit to the range [0, 1]
            digit_normalized = digit_resized / 255.0
            
            # Add the processed digit to the list
            digit_images.append(digit_normalized)
    
    return digit_images

@app.route("/")
def home():
    """
    Render the home page with links to both functionalities.
    """
    return render_template("home.html")

@app.route("/canvas")
def canvas():
    """
    Render the canvas-based digit recognition page.
    """
    return render_template("canvas.html")

@app.route("/webcam")
def webcam():
    """
    Render the webcam-based digit recognition page.
    """
    return render_template("webcam.html")

@app.route('/image-upload')
def image_upload_page():
    """Serve the image upload page."""
    return render_template('image_upload.html')

@app.route('/predict_canvas', methods=['POST'])
def predict_canvas():
    """
    Predict digits from canvas input.
    """
    data = request.json
    image_data = data['image']

    digits = preprocess_canvas_image(image_data)
    if not digits:
        return jsonify({'digits': 'No digits detected'})

    predictions = [np.argmax(model.predict(digit.reshape(1, 28, 28, 1))) for digit in digits]
    return jsonify({'digits': ''.join(map(str, predictions))})

@app.route('/predict_webcam', methods=['POST'])
def predict_webcam():
    """
    Predict digits from webcam input.
    """
    # Read the raw binary image data from the request
    image_data = request.data
    np_array = np.frombuffer(image_data, dtype=np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Preprocess the frame and predict digits
    digits = preprocess_webcam_image(frame)
    if not digits:
        return jsonify({'digits': 'No digits detected'})

    predictions = [np.argmax(model.predict(digit.reshape(1, 28, 28, 1))) for digit in digits]
    return jsonify({'digits': ''.join(map(str, predictions))})

@app.route('/predict_multiple_digits', methods=['POST'])
def predict_multiple_digits():
    """
    Predict multiple digits from an uploaded image.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Read the uploaded image
    image = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    # Preprocess the uploaded image for multiple digits
    digit_images = preprocess_uploaded_image_for_multiple_digits(image)
    
    if not digit_images:
        return jsonify({'digits': 'No digits detected'})

    # Predict digits using the pre-trained model
    predictions = [np.argmax(model.predict(digit.reshape(1, 28, 28, 1))) for digit in digit_images]
    
    # Return the predicted digits as a string
    return jsonify({'digits': ''.join(map(str, predictions))})

if __name__ == '__main__':
    app.run(debug=True)
