import base64
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for, flash, session,jsonify,make_response
from tensorflow.keras.models import load_model
from pymongo import MongoClient
import bcrypt
import secrets
from werkzeug.security import check_password_hash,generate_password_hash
from functools import wraps

app = Flask(__name__)
app.secret_key = secrets.token_hex(16) 

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["HDR"]
users_collection = db["users"]

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

    _, image = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)

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
    1. Convert the frame to grayscale.
    2. Apply thresholding to extract white-on-black digits.
    3. Find contours for each digit.
    4. Resize each digit to 28x28 pixels and normalize pixel values.
    """

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply binary inverse thresholding (digits as white on black background)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digit_images = []
    for contour in contours:
        # Get bounding box for the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Filter out small noise or irrelevant contours
        if h > 20 and w > 20:  # Adjust based on expected digit sizes
            # Extract the digit region
            digit = thresh[y:y+h, x:x+w]

            # Resize the digit to 28x28 pixels
            digit = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)

            # Normalize pixel values to range [0, 1]
            digit = digit.astype('float32') / 255.0

            # Append the digit with its x-coordinate for sorting
            digit_images.append((x, digit))

    # Sort digits by their x-coordinate (left-to-right order)
    digit_images = sorted(digit_images, key=lambda x: x[0])

    # Return only the image data (not the x-coordinates)
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
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
    
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
    Render the home page with the login form.
    """
    session.clear()
    return render_template("login.html")

# Register route
@app.route('/register', methods=['GET', 'POST'])
def register():
    try:
        if request.method == 'POST':
            # Retrieve form data
            username = request.form.get('name')
            email = request.form.get('email')
            password = request.form.get('password')
            confirm_password = request.form.get('confirm_password')  # Fixed key name

            # Validate input fields
            if not username or not password:
                flash("Username and password are required!", "error")
                return redirect(url_for('register'))

            if not email:
                flash("Email is required!", "error")
                return redirect(url_for('register'))

            if password != confirm_password:
                flash("Passwords do not match!", "error")
                return redirect(url_for('register'))

            # Check if username already exists
            if users_collection.find_one({"username": username}):
                flash("Username already exists! Please choose a different one.", "error")
                return redirect(url_for('register'))

            # Hash the password and store in the database
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())#unicode transformation format
            users_collection.insert_one({
                "username": username,
                "email": email,
                "password": hashed_password
            })

            flash("User registered successfully! Please log in.", "success")
            return redirect(url_for('login'))

        # Render the registration page
        return render_template('register.html')

    except Exception as e:
        # Log the error and display a friendly message to the user
        app.logger.error(f"Error during registration: {e}")
        flash("An unexpected error occurred. Please try again later.", "error")
        return redirect(url_for('register'))
    
@app.route('/login', methods=['POST', 'GET'])
def login():
    try:
        if request.method == 'POST':
            # Retrieve the form data
            username = request.form['username']
            password = request.form['password']

            if not username or not password:
                flash('Username and password are required.', 'error')
                return redirect(url_for('login'))  # Redirect to login if fields are empty

            # Check if the user exists in the database
            user = users_collection.find_one({'username': username})

            # If user exists and the password matches
            if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):  # Compare hashed password
                session['username'] = username  # Store the username in session
                flash('Login successful!', 'success')
                return redirect(url_for('index2'))  # Redirect to the 'index2' route after successful login

            # If the credentials are incorrect
            flash('Invalid username or password', 'error')
            return redirect(url_for('login'))  # Redirect back to the login page

        # For GET requests, render the login page
        return render_template('login.html')
    except Exception as e:
        app.logger.error(f"Error during login: {e}")  # Log the error for debugging
        flash('An unexpected error occurred. Please try again later.', 'error')
        return redirect(url_for('login'))  # Redirect back to the login page if an exception occurs


@app.route('/logout')
def logout():
    # Clear all session data
    session.clear()

    # Prevent caching after logout
    response = make_response(redirect(url_for('login')))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    
    flash("You have been logged out successfully.", "success")
    return response


@app.route('/mainpage')
def mainpage():
    return render_template("mainpage.html")

@app.route('/register_signin')
def register_signin():
    return render_template("register.html")

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:  # Check if user is logged in
            flash("You need to log in first.", "error")
            return redirect(url_for('login'))  # Redirect to login page if not logged in
        return f(*args, **kwargs)
    return decorated_function

@login_required
@app.route('/index2')
def index2():
    if 'username' in session:
        return render_template('index2.html')  # Render the index2.html template
    else:
        return redirect(url_for('login'))  # If not logged in, redirect to the login page


@app.route('/purpose')
def purpose():
    return render_template("index_purpose.html")

@app.route('/application')
def application():
    return render_template("index_applications.html")


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

@app.route('/image_upload')
def image_upload():
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

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')

if __name__ == '__main__':
    app.run(debug=True)
