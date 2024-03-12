# Project Name 
Plant Diesease Prediction 
 
## Description 
The plant disease prediction and guidance application is designed to assist users in identifying and addressing diseases affecting plants. By leveraging image recognition technology and machine learning algorithms, the app can analyze images of plant leaves and accurately predict the specific disease present. Upon predicting the disease, the application provides detailed information on the identified disease, including its common causes and recommended remedies or treatments.  
 
Users can simply upload an image of a plant leaf through the app, and within seconds, receive a diagnosis along with actionable insights on how to manage the disease effectively. This tool aims to empower gardeners, farmers, and plant enthusiasts with the knowledge and guidance needed to protect their plants from diseases and maintain their health and vitality.  
 
With a user-friendly interface and efficient prediction capabilities, the application serves as a valuable resource for individuals seeking to proactively monitor and care for their plants. Whether it's identifying fungal infections, bacterial diseases, or nutritional deficiencies, this app offers a comprehensive solution for plant disease management and cultivation success.
 
## Installation 
1. Install Python (if not already installed) 
2. Install required libraries using  pip install -r requirements.txt  
 
## Usage 
1. Run the Flask application using  python app.py  
2. Use the following endpoints: 
   -  /  - to check if the backend is running 
   -  /test  - for testing purposes 
   -  /predict  - to predict plant diseases from images 
 
## Endpoints 
-  /  - Backend is running for BTP 
-  /test  - Backend is running testing 
-  /predict  - Make predictions for plant diseases from images