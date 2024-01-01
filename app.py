from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
import tensorflow as tf
import pickle

app = Flask(__name__) 

# Load the TFLite model
model_path = 'model.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape'][1:3]

# Preprocess the image for model input
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

        # print(output_data) 

        # sorted_indices = np.argsort(output_data)[::-1]

        # print(sorted_indices)

        # indices_of_1 = np.where(sorted_indices == 1)[1][0]
        # indices_of_2 = np.where(sorted_indices == 2)[1][0]
        # indices_of_3 = np.where(sorted_indices == 3)[1][0]

        # print("Indices of 1:", indices_of_1) 
        # print(indices_of_2)    
        # print(indices_of_3)

        # Use the sorted indices to rearrange the array
        # sorted_output_data = output_data[0][sorted_indices]   

        # print(sorted_output_data)


        # Return the prediction as JSON  
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
