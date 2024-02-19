from flask import Flask, render_template, request,jsonify
import joblib
import numpy as np
import pickle

app = Flask(__name__)


# Load the pickled model and label encoders
model_rf = joblib.load('model_rf.pkl')
#model = pickle.load(open('model.pkl', 'rb'))
label_encoders = {
    'body_type': joblib.load('body_type_label_encoder.pkl'),
    'car_company': joblib.load('car_company_label_encoder.pkl'),
    'insurance': joblib.load('insurance_label_encoder.pkl'),
    'transmission_type': joblib.load('transmission_type_label_encoder.pkl'),
    'insurance': joblib.load('insurance_label_encoder.pkl'),
    'fuel_type': joblib.load('fuel_type_label_encoder.pkl'),
    'body_type': joblib.load('body_type_label_encoder.pkl'),
    'city': joblib.load('city_label_encoder.pkl'),
    'car_company': joblib.load('car_company_label_encoder.pkl'),
    'model': joblib.load('model_label_encoder.pkl'),
}

@app.route('/')
def index():
    return render_template('index.html')

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
    
    # Transform input data
            #transformed_data = []
           # for feature in ['engine_capacity','insurance','transmission_type','kms_driven','owner_type','fuel_type','seats','mileage','body_type','city','car_company','age','model' ]:
          #   transformed_data.append(label_encoders[feature].transform([globals()[feature]])[0])
           # transformed_data = np.array(transformed_data).reshape(1, -1)



            insurance_en = label_encoders['insurance'].transform([insurance])[0]
            transmission_type_en = label_encoders['transmission_type'].transform([transmission_type])[0]
            fuel_type_en = label_encoders['fuel_type'].transform([fuel_type])[0]
            body_type_en = label_encoders['body_type'].transform([body_type])[0]
            city_en= label_encoders['city'].transform([city])[0]
            car_company_en = label_encoders['car_company'].transform([car_company])[0]
            model_en = label_encoders['model'].transform([model])[0]
            
            # Make a prediction using the loaded model
            prediction_input = np.array([[engine_capacity,insurance_en, transmission_type_en, kms_driven, owner_type,fuel_type_en, seats, mileage, body_type_en, city_en,car_company_en, age,model_en]])
            predicted_price = model_rf.predict(prediction_input)[0]

          #  result = predicted_price
          #  return result

            

          #  return jsonify({'prediction': predicted_price})
    
    # Make prediction
            predicted_price = int(predicted_price)
           # result = {'prediction': predicted_price[0]}
           # prediction = "predicted_price"
           # return jsonify({'prediction':predicted_price})
            return render_template('result_new.html', predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(port=5001, debug=True)
