from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)

# Load your merged dataframe (merged_df) here
merged_df = pd.read_csv('modelling.csv')

def load_arima_model(country_name, forecast_type):
    # Load the ARIMA model from the pickle file
    model_file = f'{country_name}_{forecast_type}_arima_model.pkl'
    with open(model_file, 'rb') as file:
        model = pickle.load(file)
    return model

def train_arima_model(country_name, forecast_type):
    # Filter the DataFrame to include data only for the specified country
    country_df = merged_df[merged_df['Entity'] == country_name]

    # Convert the Pandas Series to a 1D NumPy array
    x_values = country_df['Year'].to_numpy()
    y_values = country_df[forecast_type].to_numpy()

    # Fit ARIMA model
    model = ARIMA(y_values, order=(1, 1, 1))  # You can adjust the order as needed
    model_fit = model.fit()

    # Export the trained model to a pickle file
    model_file = f'{country_name}_{forecast_type}_arima_model.pkl'
    with open(model_file, 'wb') as file:
        pickle.dump(model_fit, file)

def make_forecast(country_name, forecast_type, model):
    # Filter the DataFrame to include data only for the specified country
    country_df = merged_df[merged_df['Entity'] == country_name]

    # Forecast for future years
    forecast_steps = 5  # forecast for the next 5 years
    forecast = model.forecast(steps=forecast_steps)

    # Generate future year indices
    future_years = np.arange(country_df['Year'].max() + 1, country_df['Year'].max() + forecast_steps + 1)

    return future_years, forecast

@app.route('/')
def index():
    return render_template('powerplant.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    country_name = request.form['country_name']
    forecast_type = request.form['forecast_type']

    # Load ARIMA model
    try:
        model = load_arima_model(country_name, forecast_type)
    except FileNotFoundError:
        # Train ARIMA model if not found
        train_arima_model(country_name, forecast_type)
        # Load the newly trained model
        model = load_arima_model(country_name, forecast_type)

    # Make forecast
    forecast_years, forecast_values = make_forecast(country_name, forecast_type, model)

    # Prepare forecast data
    forecast_data = [(year, value) for year, value in zip(forecast_years, forecast_values)]

    return render_template('forecast.html', country_name=country_name, forecast_type=forecast_type, forecast_data=forecast_data)

if __name__ == '__main__':
    app.run(debug=True)
