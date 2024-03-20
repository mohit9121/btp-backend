from flask import jsonify, request
from app import app
from PIL import Image
import numpy as np
import io
import tensorflow as tf
import pickle


model_path = './static/model.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape'][1:3]

import json

# Read the contents of the JSON file
with open('./static/plant_disease_info.json', 'r') as file:
    plant_info = json.load(file)

def preprocess_image(image):
    img = image.convert("RGB")
    img = img.resize(input_shape[::-1])
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = img_array.reshape(1, *input_shape, 3)
    return img_array

def get_disease_info(disease_name):
    for disease in plant_info:
        if disease['disease'] == disease_name:
            return disease
    return "Error occured in getting information about this disease"

    

@app.route('/test')
def hello_world2():
    return 'Backend is running testing '


# Predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the request
        file = request.files['image']
        img = Image.open(file)
        
        # Preprocess the image
        input_data = preprocess_image(img)

        # Make prediction using TFLite
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
 
        plant_disease_names = ['Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_healthy', 'Blueberry_healthy', 'Cherry(including_sour)__Powdery_mildew', 'Cherry(including_sour)__healthy', 'Corn(maize)__Cercospora_leaf_spot Gray_leaf_spot', 'Corn(maize)__Common_rust', 'Corn_(maize)__Northern_Leaf_Blight', 'Corn(maize)__healthy', 'Grape_Black_rot', 'Grape_Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 'Grape__healthy', 'Orange_Haunglongbing(Citrus_greening)', 'Peach__Bacterial_spot', 'Peach_healthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,_bell_healthy', 'Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy', 'Raspberry_healthy', 'Soybean_healthy', 'Squash_Powdery_mildew', 'Strawberry_Leaf_scorch', 'Strawberry_healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites Two-spotted_spider_mite', 'Tomato_Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Tomato_mosaic_virus', 'Tomato__healthy']

        sorted_indices = np.argsort(output_data[0])[::-1]
        top_three_indices = sorted_indices[:5] 
        print(top_three_indices)

        predicted_class = plant_disease_names[np.argmax(output_data)] 
        print("all possible disease are: ") 
        print(plant_disease_names[top_three_indices[0]])

        disease_info = get_disease_info(predicted_class)
        print(disease_info)

        # return jsonify({'prediction': predicted_class}) 
        return disease_info
 
    except Exception as e:
        return jsonify({'error': str(e)}), 500
