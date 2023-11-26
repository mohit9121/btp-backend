from flask import Flask, request, jsonify
# import tensorflow as tf
import tensorflow.keras as keras
import h5py
# from keras.models import Model
import numpy as np
from PIL import Image
import io

app = Flask(__name__) 

# Load your pre-trained .h5 model

model = keras.models.load_model('./Prototype-Green_Hackathon_model.h5')

filename = 'requirements.txt'

infile = open(filename, 'r') 

print(infile.readline())
# with open(filename) as f:
#     requirements = f.read().splitlines()

# print(requirements)

class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

@app.route('/')
def hello_world():
    return 'Backend is running. BTP of Mohit Anirudh and Aman IIT Ropar'

@app.route('/test')
def hello_world2():
    return 'Backend is running testing '


# Define a route for disease prediction
@app.route('/predict', methods=['POST'])
def predict_disease():
    try:
        # Get the image from the request
        image = request.files['image']
        image = Image.open(io.BytesIO(image.read()))

        image = image.resize((100, 100))

        image = np.array(image)

        # Preprocess the image (resize, normalize, etc.)
        # You may need to adapt this preprocessing based on your model's requirements



        # Make a prediction
        prediction = model.predict(np.expand_dims(image, axis=0))

        prediction = prediction[0]

        print(class_name[np.argmax(prediction)])

        # You may need to post-process the prediction as per your model's output
        # Convert the prediction to human-readable format

        return jsonify({'prediction': class_name[np.argmax(prediction)]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()


# ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency con
# flicts.
# scipy 1.7.3 requires numpy<1.23.0,>=1.16.5, but you have numpy 1.26.1 which is incompatible.
# numba 0.55.1 requires numpy<1.22,>=1.18, but you have numpy 1.26.1 which is incompatible.
# pd.read_hdf(r"\home\fbarajas\newsite\data\YAHOO.H5","TWTR")
# tensorflow==2.11.0
# Flask==2.2.5
# numpy==1.21.6
# Pillow==9.2.0