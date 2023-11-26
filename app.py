from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
import tensorflow as tf

app = Flask(__name__) 

# model = tf.keras.models.load_model('./Prototype-Green_Hackathon_model.h5')

# class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

@app.route('/')
def hello_world():
    return 'Backend is running. BTP of Mohit Anirudh and Aman IIT Ropar'

@app.route('/test')
def hello_world2():
    return 'Backend is running testing '

# Define a route for disease prediction
# @app.route('/predict', methods=['POST'])
# def predict_disease():
#     try:
#         image = request.files['image']
#         image = Image.open(io.BytesIO(image.read()))
#         image = image.resize((100, 100))
#         image = np.array(image)
#         output = 'potato_unhealthy'
#         prediction = model.predict(np.expand_dims(image, axis=0))
#         prediction = prediction[0]
#         print(class_name[np.argmax(prediction)])
#         output = class_name[np.argmax(prediction)]
#         return jsonify({'prediction': output})
#     except Exception as e:
#         return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()
