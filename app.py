from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
import tensorflow as tf
import pickle

app = Flask(__name__) 

model_path = 'model.tflite'
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

@app.route('/')
def hello_world():
    return 'Backend is running for BTP'

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
        # print(plant_disease_names[top_three_indices[1]]) 
        # print(plant_disease_names[top_three_indices[2]]) 
        # print(plant_disease_names[top_three_indices[3]]) 
        # print(plant_disease_names[top_three_indices[4]]) 
        return jsonify({'prediction': predicted_class}) 
 
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict1', methods=['POST'])
def predict_disease1():
    return 'post request response'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
