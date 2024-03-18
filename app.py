from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
import tensorflow as tf
import pickle

app = Flask(__name__) 

model_path = './static/model.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape'][1:3]



def preprocess_image(image):
    img = image.convert("RGB")
    img = img.resize(input_shape[::-1])
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = img_array.reshape(1, *input_shape, 3)
    return img_array

# class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

import json

# Read the contents of the JSON file
with open('./static/plant_disease_info.json', 'r') as file:
    plant_info = json.load(file)

# Access the data
for disease in plant_info:
    # print("Disease:", disease['disease'])
    # print("Causes:", disease['causes'])
    # print("Remedies:", disease['remedies'])
    print()


def get_disease_info(disease_name): 
    for disease in plant_info:
        if str(disease['disease']) == str(disease_name):
            return disease
    return "Error occured in getting information about this disease"

    

@app.route('/')
def hello_world():
    return 'Backend is running for BTP'



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
 
        # plant_disease_names = ['Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_healthy', 'Blueberry_healthy', 'Cherry(including_sour)__Powdery_mildew', 'Cherry(including_sour)__healthy', 'Corn(maize)__Cercospora_leaf_spot Gray_leaf_spot', 'Corn(maize)__Common_rust', 'Corn_(maize)__Northern_Leaf_Blight', 'Corn(maize)__healthy', 'Grape_Black_rot', 'Grape_Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 'Grape__healthy', 'Orange_Haunglongbing(Citrus_greening)', 'Peach__Bacterial_spot', 'Peach_healthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,_bell_healthy', 'Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy', 'Raspberry_healthy', 'Soybean_healthy', 'Squash_Powdery_mildew', 'Strawberry_Leaf_scorch', 'Strawberry_healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites Two-spotted_spider_mite', 'Tomato_Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Tomato_mosaic_virus', 'Tomato__healthy']
        plant_disease_names = [
    'Apple___Apple_scab', 
    'Apple___Black_rot', 
    'Apple___Cedar_apple_rust', 
    'Apple___healthy', 
    'Blueberry___healthy', 
    'Cherry_(including_sour)___Powdery_mildew', 
    'Cherry_(including_sour)___healthy', 
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
    'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 
    'Corn_(maize)___healthy', 
    'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
    'Grape___healthy', 
    'Orange___Haunglongbing_(Citrus_greening)', 
    'Peach___Bacterial_spot', 
    'Peach___healthy', 
    'Pepper,_bell___Bacterial_spot', 
    'Pepper,_bell___healthy', 
    'Potato___Early_blight', 
    'Potato___Late_blight', 
    'Potato___healthy', 
    'Raspberry___healthy', 
    'Soybean___healthy', 
    'Squash___Powdery_mildew', 
    'Strawberry___Leaf_scorch', 
    'Strawberry___healthy', 
    'Tomato___Bacterial_spot', 
    'Tomato___Early_blight', 
    'Tomato___Late_blight', 
    'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 
    'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
    'Tomato___Tomato_mosaic_virus', 
    'Tomato___healthy'
]

        sorted_indices = np.argsort(output_data[0])[::-1]
        top_three_indices = sorted_indices[:5] 
        print(top_three_indices)

        predicted_class = plant_disease_names[np.argmax(output_data)] 
        print("possible disease: ") 
        print(plant_disease_names[top_three_indices[0]])

        disease_info = get_disease_info(predicted_class)
        print(disease_info)

        return disease_info  
 
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict1', methods=['POST'])
def predict_disease1():
    return 'post request response'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
