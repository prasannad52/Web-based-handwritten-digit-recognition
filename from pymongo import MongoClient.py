from pymongo import MongoClient
import bcrypt

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")  # Update with your MongoDB URI if needed
db = client["HDR"]  # Replace with your database name
users_collection = db["users"]  # Replace with your collection name

# Function to hash a password
def hash_password(password):
    # Generate a salt and hash the password
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed

# Function to verify a password
def verify_password(input_password, stored_password):
    # Compare the input password with the stored hash
    return bcrypt.checkpw(input_password.encode('utf-8'), stored_password)

# Function to create a new user
def create_user(username, password):
    # Check if the username already exists
    if users_collection.find_one({"username": username}):
        return "Username already exists."

    # Hash the password and insert the user into the database
    hashed_password = hash_password(password)
    user = {"username": username, "password": hashed_password}
    users_collection.insert_one(user)
    return "User created successfully."

# Function to login a user
def login_user(username, password):
    # Find the user in the database
    user = users_collection.find_one({"username": username})
    if user and verify_password(password, user["password"]):
        return "Login successful!"
    return "Invalid username or password."

# Main program loop for testing
if __name__ == "__main__":
    while True:
        print("\n=== Login System ===")
        print("1. Register")
        print("2. Login")
        print("3. Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            username = input("Enter a username: ")
            password = input("Enter a password: ")
            message = create_user(username, password)
            print(message)

        elif choice == "2":
            username = input("Enter your username: ")
            password = input("Enter your password: ")
            message = login_user(username, password)
            print(message)

        elif choice == "3":
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please try again.")
