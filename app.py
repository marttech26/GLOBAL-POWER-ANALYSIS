import streamlit as st
import numpy as np
import pandas as pd
import pickle
from statsmodels.tsa.arima.model import ARIMA

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

def index():
    st.title('GLOBAL POWER ANALYSIS')
    country_name = st.selectbox('Select country:', ['Afghanistan', 'Albania', 'Algeria', 'American Samoa', 'Angola',
       'Antigua and Barbuda', 'Argentina', 'Armenia', 'Aruba',
       'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain',
       'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin',
       'Bermuda', 'Bhutan', 'Bolivia', 'Bosnia and Herzegovina',
       'Botswana', 'Brazil', 'British Virgin Islands', 'Brunei',
       'Bulgaria', 'Burkina Faso', 'Burundi', 'Cambodia', 'Cameroon',
       'Canada', 'Cape Verde', 'Cayman Islands',
       'Central African Republic', 'Chad', 'Chile', 'China', 'Colombia',
       'Comoros', 'Congo', 'Costa Rica', "Cote d'Ivoire", 'Croatia',
       'Cuba', 'Cyprus', 'Czechia', 'Democratic Republic of Congo',
       'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic',
       'East Timor', 'Ecuador', 'Egypt', 'El Salvador',
       'Equatorial Guinea', 'Eritrea', 'Estonia', 'Eswatini', 'Ethiopia',
       'European Union (27)', 'Fiji', 'Finland', 'France',
       'French Polynesia', 'Gabon', 'Gambia', 'Georgia', 'Germany',
       'Ghana', 'Gibraltar', 'Greece', 'Greenland', 'Grenada', 'Guam',
       'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti',
       'High-income countries', 'Honduras', 'Hong Kong', 'Hungary',
       'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland',
       'Israel', 'Italy', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan',
       'Kenya', 'Kiribati', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Latvia',
       'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Lithuania',
       'Low-income countries', 'Lower-middle-income countries',
       'Luxembourg', 'Macao', 'Madagascar', 'Malawi', 'Malaysia',
       'Maldives', 'Mali', 'Malta', 'Mauritania', 'Mauritius', 'Mexico',
       'Moldova', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique',
       'Myanmar', 'Namibia', 'Nauru', 'Nepal', 'Netherlands',
       'New Caledonia', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria',
       'North Korea', 'North Macedonia', 'Norway', 'Oman', 'Pakistan',
       'Palestine', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru',
       'Philippines', 'Poland', 'Portugal', 'Puerto Rico', 'Qatar',
       'Romania', 'Russia', 'Rwanda', 'Saint Kitts and Nevis',
       'Saint Lucia', 'Saint Vincent and the Grenadines', 'Samoa',
       'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia',
       'Seychelles', 'Sierra Leone', 'Singapore', 'Slovakia', 'Slovenia',
       'Solomon Islands', 'Somalia', 'South Africa', 'South Korea',
       'South Sudan', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Sweden',
       'Switzerland', 'Syria', 'Tajikistan', 'Tanzania', 'Thailand',
       'Togo', 'Tonga', 'Trinidad and Tobago', 'Tunisia', 'Turkey',
       'Turkmenistan', 'Turks and Caicos Islands', 'Uganda', 'Ukraine',
       'United Arab Emirates', 'United Kingdom', 'United States',
       'United States Virgin Islands', 'Upper-middle-income countries',
       'Uruguay', 'Uzbekistan', 'Vanuatu', 'Venezuela', 'Vietnam',
       'World', 'Yemen', 'Zambia', 'Zimbabwe'])
    forecast_type = st.selectbox('Select forecast type:', ['Electricity generation - TWh', 
                                                          'Electricity from hydro - TWh', 
                                                          'Electricity from solar - TWh', 
                                                          'Other renewables including bioenergy - TWh', 
                                                          'Electricity from wind - TWh', 
                                                          'Electricity from Non-Renewables - TWh',
                                                          'Total Renewable Electricity - TWh'])
    generate_button = st.button('Generate Forecast')
    return country_name, forecast_type, generate_button


def forecast(country_name, forecast_type):
    try:
        model = load_arima_model(country_name, forecast_type)
    except FileNotFoundError:
        train_arima_model(country_name, forecast_type)
        model = load_arima_model(country_name, forecast_type)

    forecast_years, forecast_values = make_forecast(country_name, forecast_type, model)
    forecast_data = [(year, value) for year, value in zip(forecast_years, forecast_values)]
    return forecast_data

# Function to display forecast in a separate page
def show_forecast_page(forecast_data, country_name, forecast_type):
    # Clear the main page
    st.empty()

    # Header for the forecast page
    st.header(f'Forecast for {country_name} ({forecast_type}):')

    # Display the forecast as a table
    forecast_df = pd.DataFrame(forecast_data, columns=['Year', 'Forecast'])
    st.table(forecast_df)

def main():
    st.title('GLOBAL POWER ANALYSIS')

    # Sidebar for user input
    country_name = st.sidebar.selectbox('Select country:', ['Afghanistan', 'Albania', 'Algeria', 'American Samoa', 'Angola',
       'Antigua and Barbuda', 'Argentina', 'Armenia', 'Aruba',
       'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain',
       'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin',
       'Bermuda', 'Bhutan', 'Bolivia', 'Bosnia and Herzegovina',
       'Botswana', 'Brazil', 'British Virgin Islands', 'Brunei',
       'Bulgaria', 'Burkina Faso', 'Burundi', 'Cambodia', 'Cameroon',
       'Canada', 'Cape Verde', 'Cayman Islands',
       'Central African Republic', 'Chad', 'Chile', 'China', 'Colombia',
       'Comoros', 'Congo', 'Costa Rica', "Cote d'Ivoire", 'Croatia',
       'Cuba', 'Cyprus', 'Czechia', 'Democratic Republic of Congo',
       'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic',
       'East Timor', 'Ecuador', 'Egypt', 'El Salvador',
       'Equatorial Guinea', 'Eritrea', 'Estonia', 'Eswatini', 'Ethiopia',
       'European Union (27)', 'Fiji', 'Finland', 'France',
       'French Polynesia', 'Gabon', 'Gambia', 'Georgia', 'Germany',
       'Ghana', 'Gibraltar', 'Greece', 'Greenland', 'Grenada', 'Guam',
       'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti',
       'High-income countries', 'Honduras', 'Hong Kong', 'Hungary',
       'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland',
       'Israel', 'Italy', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan',
       'Kenya', 'Kiribati', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Latvia',
       'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Lithuania',
       'Low-income countries', 'Lower-middle-income countries',
       'Luxembourg', 'Macao', 'Madagascar', 'Malawi', 'Malaysia',
       'Maldives', 'Mali', 'Malta', 'Mauritania', 'Mauritius', 'Mexico',
       'Moldova', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique',
       'Myanmar', 'Namibia', 'Nauru', 'Nepal', 'Netherlands',
       'New Caledonia', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria',
       'North Korea', 'North Macedonia', 'Norway', 'Oman', 'Pakistan',
       'Palestine', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru',
       'Philippines', 'Poland', 'Portugal', 'Puerto Rico', 'Qatar',
       'Romania', 'Russia', 'Rwanda', 'Saint Kitts and Nevis',
       'Saint Lucia', 'Saint Vincent and the Grenadines', 'Samoa',
       'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia',
       'Seychelles', 'Sierra Leone', 'Singapore', 'Slovakia', 'Slovenia',
       'Solomon Islands', 'Somalia', 'South Africa', 'South Korea',
       'South Sudan', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Sweden',
       'Switzerland', 'Syria', 'Tajikistan', 'Tanzania', 'Thailand',
       'Togo', 'Tonga', 'Trinidad and Tobago', 'Tunisia', 'Turkey',
       'Turkmenistan', 'Turks and Caicos Islands', 'Uganda', 'Ukraine',
       'United Arab Emirates', 'United Kingdom', 'United States',
       'United States Virgin Islands', 'Upper-middle-income countries',
       'Uruguay', 'Uzbekistan', 'Vanuatu', 'Venezuela', 'Vietnam',
       'World', 'Yemen', 'Zambia', 'Zimbabwe'])  # All entries in merged df
    forecast_type = st.sidebar.selectbox('Select forecast type:', ['Electricity generation - TWh', 
                                                          'Electricity from hydro - TWh', 
                                                          'Electricity from solar - TWh', 
                                                          'Other renewables including bioenergy - TWh', 
                                                          'Electricity from wind - TWh', 
                                                          'Electricity from Non-Renewables - TWh',
                                                          'Total Renewable Electricity - TWh'])

    # Main page button
    generate_button = st.button('Generate Forecast')

    # Conditional rendering based on button click
    if generate_button:
        forecast_data = forecast(country_name, forecast_type)
        # Save the forecast data in session state to maintain state after reruns
        st.session_state.forecast_data = forecast_data
        st.session_state.country_name = country_name
        st.session_state.forecast_type = forecast_type
        # Redirect to forecast page
        show_forecast_page(forecast_data, country_name, forecast_type)
    elif 'forecast_data' in st.session_state:
        # If session state exists, show the forecast page
        show_forecast_page(st.session_state.forecast_data, st.session_state.country_name, st.session_state.forecast_type)

if __name__ == '__main__':
    st.set_page_config(page_title='GLOBAL POWER ANALYSIS', layout='wide')
    main()
