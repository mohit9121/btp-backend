from flask import Flask

# Initialize the Flask application instance
app = Flask(__name__)

# Import routes from the routes module
from app import routes
