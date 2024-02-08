from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained model
model = pickle.load('model.pkl', 'rb')

# Define a route for the home page
@app.route('/')
def home():
    return render_template('powerplant.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    input_features =[float(request.form.get('Electricity from wind - TWh')),
                  float(request.form.get('Electricity from hydro - TWh')),
                  float(request.form.get('Electricity from solar - TWh')),
                  float(request.form.get('Other renewables including bioenergy - TWh'))
    ]

    # Make prediction
    prediction = model.predict([input_features])[0]

    # Return prediction as JSON
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
