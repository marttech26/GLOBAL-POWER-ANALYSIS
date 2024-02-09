from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np  # Import NumPy for array manipulation
import pandas as pd  # Import Pandas for data manipulation

app = Flask(__name__)

# Load pickled models
models = {}
model_files = [
    "pipeline_wind.pkl", 
    "pipeline_hydro.pkl", 
    "pipeline_solar.pkl", 
    "pipeline_bioenergy.pkl"  
]

for file in model_files:
    with open(file, 'rb') as f:
        models[file.split('_')[1].split('.')[0]] = pickle.load(f)  

# Load country data from CSV file
countries_df = pd.read_csv('modelling.csv')
countries = countries_df['Entity'].tolist()

@app.route('/')
def index():
    return render_template('powerplant.html', countries=countries)

@app.route('/predict', methods=['POST'])
def predict():
    model_name = request.form['model']  # Get selected model from form
    year = request.form['year']  # Get year from form
    
    # Handle empty input fields
    wind = request.form['wind']
    hydro = request.form['hydro']
    solar = request.form['solar']
    bioenergy = request.form['bioenergy']
    entity = request.form['entity']
    total_renewable = request.form['total_renewable']
    non_renewable = request.form['non_renewable']
    
    # Convert to float if not empty, otherwise use default value of 0.0
    wind = float(wind) if wind else 0.0
    hydro = float(hydro) if hydro else 0.0
    solar = float(solar) if solar else 0.0
    bioenergy = float(bioenergy) if bioenergy else 0.0
    total_renewable = float(total_renewable) if total_renewable else 0.0
    non_renewable = float(non_renewable) if non_renewable else 0.0
    
    features = {
        'Entity': entity,
        'Electricity generation - TWh': 0,  # Placeholder value, assuming it's not needed for prediction
        'Electricity from wind - TWh': wind,
        'Electricity from hydro - TWh': hydro,
        'Electricity from solar - TWh': solar,
        'Other renewables including bioenergy - TWh': bioenergy,
        'Total Renewable Electricity - TWh': total_renewable,
        'Electricity from Non-Renewables - TWh': non_renewable,
        'Year': float(year)  # Convert to float
    }
    
    # Convert features into a 2D array with a single row
    X = np.array([list(features.values())])  
    
    # Convert features into a DataFrame
    X_df = pd.DataFrame(data=X, columns=features.keys())
    
    # Predict using the selected model
    pipeline = models[model_name]
    prediction = pipeline.predict(X_df)
    
    return jsonify({'prediction': prediction.tolist()})


if __name__ == '__main__':
    app.run(debug=True)
