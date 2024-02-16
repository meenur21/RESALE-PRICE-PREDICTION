from flask import Flask, render_template, request, jsonify
#import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import joblib
import pickle
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)

# Load the trained model
model_rf = pickle.load(open('model_rf.pkl', 'rb'))
#model = joblib.load('d_model.joblib')

# model = joblib.load('d_model.joblib')

label_encoders = LabelEncoder()

# Render the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Handle the form submission and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    
            # Get user input from the form
            engine_capacity = int(request.form['engine_capacity'])
            insurance = request.form['insurance']
            transmission_type = request.form['transmission_type']
            kms_driven = float(request.form['kms_driven'])
            owner_type = float(request.form['owner_type'])
            fuel_type = request.form['fuel_type']
            seats = float(request.form['seats'])
            mileage = float(request.form['mileage'])
            body_type = request.form['body_type']
            city = request.form['city']
            car_company = request.form['car_company']
            age = float(request.form['age'])
            model = request.form['model']
            
            #engine_capacity_en = label_encoders['engine_capacity'].transform([engine_capacity])[0]
            insurance_en = label_encoders['insurance'].transform([insurance])[0]
            transmission_type_en = label_encoders['transmission_type'].transform([transmission_type])[0]
            fuel_type_en = label_encoders['fuel_type'].transform([fuel_type])[0]
            body_type_en = label_encoders['body_type'].transform([body_type])[0]
            city_en= label_encoders['city'].transform([city])[0]
            car_company_en = label_encoders['car_company'].transform([car_company])[0]
            model_en = label_encoders['model'].transform([model])[0]
            
            # Make a prediction using the loaded model
            prediction_input = np.array([[ engine_capacity,insurance_en, transmission_type_en, kms_driven, owner_type,fuel_type_en, seats, mileage, body_type_en, city_en,car_company_en, age,model_en]])
            predicted_price = model_rf.predict(prediction_input)

            # Return the predicted species
            result = {'prediction': predicted_price[0]}
            return render_template('result.html', result=result)

       

if __name__ == '__main__':
    app.run(debug=True)