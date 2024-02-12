# Importing necessary libraries for data manipulation and visualization
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import datetime as dt
import math

# Importing machine learning modules
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from statsmodels.tsa.arima.model import ARIMA


def univariate_analysis_histograms(dataframe):
    """
    Perform univariate analysis using histograms for numerical columns in the dataframe.

    Parameters:
    dataframe (pandas DataFrame): The input dataframe for analysis.

    Returns:
    None
    """
    # Exclude non-numeric columns
    num_cols = dataframe.columns.difference(['Entity', 'Year'])
    num_plots = len(num_cols)

    # Create subplots
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))

    # Plot histograms for each numerical column
    for i, col in enumerate(num_cols):
        row_index = i // (num_plots // 2 + num_plots % 2)
        col_index = i % (num_plots // 2 + num_plots % 2)
        dataframe[col].hist(ax=axes[row_index, col_index], bins=20, color='skyblue', alpha=0.7)
        axes[row_index, col_index].set_title(f'Distribution of {col}')
        axes[row_index, col_index].set_xlabel(col)
        axes[row_index, col_index].set_ylabel('Frequency')

    plt.tight_layout(pad=1.0)
    plt.show()



def plot_violin_plots(dataframe, columns, figsize=(14, 10)):
    """
    Generate violin plots for specified columns in the dataframe.

    Parameters:
    dataframe (pandas DataFrame): The input dataframe.
    columns (list): List of column names to plot.
    figsize (tuple, optional): Figure size for the plots. Defaults to (14, 10).

    Returns:
    None
    """
    plt.figure(figsize=figsize)  # Adjust the figure size

    # Create subplots
    for i, col in enumerate(columns, 1):
        plt.subplot(3, 3, i)
        sns.violinplot(x=dataframe[col], color='skyblue')
        plt.title(f'Violin plot of {col}')
        plt.xlabel(col)
        plt.tight_layout()

    plt.show()



def plot_electricity_generation(dataframe):
    """
    Generate a line plot comparing total renewable electricity generation with electricity 
    generation from non-renewable sources over the years.

    Parameters:
    dataframe (pandas DataFrame): The input dataframe containing relevant columns.

    Returns:
    None
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=dataframe, x='Year', y='Total Renewable Electricity - TWh', label='Total Renewable Electricity')
    sns.lineplot(data=dataframe, x='Year', y='Electricity from Non-Renewables - TWh', label='Electricity from Non-Renewables - TWh')
    plt.title('Total Renewable vs. Non-Renewable Electricity Generation Over Years')
    plt.xlabel('Year')
    plt.ylabel('Electricity Generation (TWh)')
    plt.legend()
    plt.show()

def plot_renewable_vs_non_renewable(dataframe):
    """
    Calculate the sum of renewable and non-renewable energy generation from the dataframe
    and plot them in a bar chart.

    Parameters:
    dataframe (pandas DataFrame): The input dataframe containing relevant columns.

    Returns:
    None
    """
    # Calculate the sum of renewable and non-renewable energy generation
    total_renewable = dataframe['Total Renewable Electricity - TWh'].sum()
    total_non_renewable = dataframe['Electricity from Non-Renewables - TWh'].sum()

    # Plot renewable vs. non-renewable energy generation
    plt.figure(figsize=(8, 5))
    plt.bar(['Renewable', 'Non-Renewable'], [total_renewable, total_non_renewable], color=['green', 'maroon'])
    plt.title('Renewable vs. Non-Renewable Energy Generation')
    plt.xlabel('Energy Source')
    plt.ylabel('Total Electricity Generation (TWh)')
    plt.show()

def plot_urbanization_rate(dataframe):
    """
    Generate a line plot showing the urbanization rate over the years.

    Parameters:
    dataframe (pandas DataFrame): The input dataframe containing relevant columns.

    Returns:
    None
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=dataframe, x='Year', y='Urbanization', label='Urbanization')
    plt.title('Urbanization Rate')
    plt.xlabel('Year')
    plt.ylabel('Urbanization Rate')
    plt.legend()
    plt.show()

def plot_top_countries_by_energy_generation(merged_df, energy_type):
    """
    Filter out specific entities, group data by entity, calculate the total energy generation, 
    and plot the top 10 countries by renewable or non-renewable energy generation.

    Parameters:
    merged_df (pandas DataFrame): The input dataframe containing relevant columns.
    energy_type (str): Type of energy generation to plot ('renewable' or 'non-renewable').

    Returns:
    None
    """
    # Filter out entities 'World', 'High income', and 'EU'
    filtered_df = merged_df[~merged_df['Entity'].isin(['World', 'High-income countries',
                                                       'Upper-middle-income countries', 
                                                       'Lower-middle-income countries',
                                                       'European Union (27)',])]

    # Group data by entity (country) and calculate the total energy generation
    if energy_type == 'renewable':
        energy_column = 'Total Renewable Electricity - TWh'
    elif energy_type == 'non-renewable':
        energy_column = 'Electricity from Non-Renewables - TWh'
    else:
        print("Invalid energy type. Please provide 'renewable' or 'non-renewable'.")
        return

    energy_by_entity = filtered_df.groupby('Entity')[energy_column].sum().sort_values(ascending=False)

    # Plot top 10 countries with the highest energy generation
    top_energy_countries = energy_by_entity.head(10)
    top_energy_countries.plot(kind='bar', figsize=(10, 6), color='skyblue')
    plt.title(f'Top 10 Countries by {energy_type.capitalize()} Energy Generation')
    plt.xlabel('Country')
    plt.ylabel(f'Total {energy_type.capitalize()} Electricity (TWh)')
    plt.xticks(rotation=90)
    plt.show()

def plot_renewable_generation_over_time(merged_df, entity):
    """
    Filter the DataFrame to include data only for a specific entity, group data by year, 
    calculate the sum of renewable energy generation, and plot renewable energy generation over time.

    Parameters:
    merged_df (pandas DataFrame): The input dataframe containing relevant columns.
    entity (str): The entity (e.g., country) for which to plot renewable energy generation over time.

    Returns:
    None
    """
    # Filter the DataFrame to include data only for the specified entity
    entity_df = merged_df[merged_df['Entity'] == entity]

    # Group data by year and calculate the sum of renewable energy generation
    renewable_generation = entity_df.groupby('Year')[['Electricity from wind - TWh', 'Electricity from hydro - TWh',
                                               'Electricity from solar - TWh', 'Other renewables including bioenergy - TWh']].sum()

    # Plot renewable energy generation over time
    renewable_generation.plot(kind='line', figsize=(10, 6))
    plt.title(f'Renewable Energy Generation in {entity} Over Time')
    plt.xlabel('Year')
    plt.ylabel('Electricity Generation (TWh)')
    plt.show()

def run_pipelines(modelling, target_column, random_state=42):
    # Define preprocessing steps for the pipeline (common for all)
    preprocessor = ColumnTransformer(
        transformers=[
            ('encoder', OneHotEncoder(), ['Entity'])
        ],
        remainder='passthrough'
    )
    
    # Filter the dataframe based on the year
    train_data = modelling[modelling['Year'] <= 2016]
    test_data = modelling[modelling['Year'] >= 2017]

    # Select relevant columns for X and y (assuming all other columns are required for features)
    feature_columns = [col for col in modelling.columns if col not in ['Year', target_column]]
    X_train = train_data[feature_columns]
    y_train = train_data[target_column]

    X_test = test_data[feature_columns]
    y_test = test_data[target_column]
    
    # Create pipelines for different regressors
    pipelines = [
        Pipeline([('preprocessor', preprocessor), ('regressor', LinearRegression())]),
        Pipeline([('preprocessor', preprocessor), ('regressor', RandomForestRegressor(random_state=random_state))]),
        Pipeline([('preprocessor', preprocessor), ('regressor', GradientBoostingRegressor(random_state=random_state))])
    ]
    
    # Fit and evaluate each model
    for pipeline in pipelines:
        pipeline.fit(X_train, y_train)
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        
        # Calculate error metrics for training and testing sets
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        # Print error metrics
        print(f'\nModel: {pipeline.named_steps["regressor"].__class__.__name__}')
        print('Training Set Metrics:')
        print(f'Mean Squared Error (MSE): {train_mse}')
        print(f'Mean Absolute Error (MAE): {train_mae}')
        print(f'R-squared (R2) Score: {train_r2}')
        print('\nTesting Set Metrics:')
        print(f'Mean Squared Error (MSE): {test_mse}')
        print(f'Mean Absolute Error (MAE): {test_mae}')
        print(f'R-squared (R2) Score: {test_r2}')

def run_pipelines_with_cross_validation(modelling, target_column, random_state=42, n_folds=5):
    """
    Perform k-fold cross-validation for each pipeline and print the error metrics.

    Parameters:
    modelling (DataFrame): The input dataframe containing relevant columns.
    target_column (str): The target column to predict.
    random_state (int, optional): Random state for reproducibility. Defaults to 42.
    n_folds (int, optional): Number of folds for cross-validation. Defaults to 5.

    Returns:
    None
    """
    # Define preprocessing steps for the pipeline (common for all)
    preprocessor = ColumnTransformer(
        transformers=[
            ('encoder', OneHotEncoder(), ['Entity'])
        ],
        remainder='passthrough'
    )

    # Filter the dataframe based on the year
    train_data = modelling[modelling['Year'] <= 2016]
    test_data = modelling[modelling['Year'] >= 2017]

    # Select relevant columns for X and y (assuming all other columns are required for features)
    feature_columns = [col for col in modelling.columns if col not in ['Year', target_column]]
    X_train = train_data[feature_columns]
    y_train = train_data[target_column]

    X_test = test_data[feature_columns]
    y_test = test_data[target_column]

    # Create pipelines for different regressors
    pipelines = [
        Pipeline([('preprocessor', preprocessor), ('regressor', LinearRegression())]),
        Pipeline([('preprocessor', preprocessor), ('regressor', RandomForestRegressor(random_state=random_state))]),
        Pipeline([('preprocessor', preprocessor), ('regressor', GradientBoostingRegressor(random_state=random_state))])
    ]

    # Define KFold cross-validator
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # Fit and evaluate each model using k-fold cross-validation
    for pipeline in pipelines:
        # Perform cross-validation
        y_pred = cross_val_predict(pipeline, X_train, y_train, cv=kf)

        # Calculate error metrics
        mse = mean_squared_error(y_train, y_pred)
        mae = mean_absolute_error(y_train, y_pred)
        r2 = r2_score(y_train, y_pred)

        # Print error metrics
        print(f'\nModel: {pipeline.named_steps["regressor"].__class__.__name__}')
        print(f'Cross-Validation MSE: {mse}')
        print(f'Cross-Validation MAE: {mae}')
        print(f'Cross-Validation R-squared (R2) Score: {r2}')


def get_predicted_actual_dataframes(modelling, target_column, random_state=42, n_folds=5):
    """
    Perform k-fold cross-validation for each pipeline and return DataFrames for predicted and actual values for both
    the training and testing sets.

    Parameters:
    modelling (DataFrame): The input dataframe containing relevant columns.
    target_column (str): The target column to predict.
    random_state (int, optional): Random state for reproducibility. Defaults to 42.
    n_folds (int, optional): Number of folds for cross-validation. Defaults to 5.

    Returns:
    train_results (DataFrame): DataFrame containing predicted and actual values for the training set.
    test_results (DataFrame): DataFrame containing predicted and actual values for the testing set.
    """
    # Define preprocessing steps for the pipeline (common for all)
    preprocessor = ColumnTransformer(
        transformers=[
            ('encoder', OneHotEncoder(), ['Entity'])
        ],
        remainder='passthrough'
    )

    # Filter the dataframe based on the year
    train_data = modelling[modelling['Year'] <= 2016]
    test_data = modelling[modelling['Year'] >= 2017]

    # Select relevant columns for X and y (assuming all other columns are required for features)
    feature_columns = [col for col in modelling.columns if col not in ['Year', target_column]]
    X_train = train_data[feature_columns]
    y_train = train_data[target_column]

    X_test = test_data[feature_columns]
    y_test = test_data[target_column]

    # Create pipelines for different regressors
    pipelines = [
        Pipeline([('preprocessor', preprocessor), ('regressor', LinearRegression())]),
        Pipeline([('preprocessor', preprocessor), ('regressor', RandomForestRegressor(random_state=random_state))]),
        Pipeline([('preprocessor', preprocessor), ('regressor', GradientBoostingRegressor(random_state=random_state))])
    ]

    # Define KFold cross-validator
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    train_results = []
    test_results = []

    # Fit and evaluate each model using k-fold cross-validation
    for pipeline in pipelines:
        # Perform cross-validation
        y_pred_train = cross_val_predict(pipeline, X_train, y_train, cv=kf)
        y_pred_test = pipeline.fit(X_train, y_train).predict(X_test)

        # Create DataFrames for predicted and actual values
        train_results.append(pd.DataFrame({'Actual': y_train, 'Predicted': y_pred_train}))
        test_results.append(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_test}))

    return train_results, test_results

def plot_renewable_energy_forecast(df, country_name):
    """
    Plot the actual and forecasted renewable energy production for a specific country.

    Parameters:
    df (DataFrame): The input dataframe containing relevant columns.
    country_name (str): The name of the country for which the forecast is to be generated.

    Returns:
    None
    """
    # Filter the DataFrame to include data only for the specified country
    country_df = df[df['Entity'] == country_name]

    # Convert the Pandas Series to a 1D NumPy array before plotting
    x_values = country_df['Year'].to_numpy()
    y_values = country_df['Total Renewable Electricity - TWh'].to_numpy()

    # Fit ARIMA model
    model = ARIMA(y_values, order=(1, 1, 1))
    model_fit = model.fit()

    # Forecast for future years
    forecast_steps = 5
    forecast = model_fit.forecast(steps=forecast_steps)

    # Generate future year indices
    future_years = np.arange(country_df['Year'].max() + 1, country_df['Year'].max() + forecast_steps + 1)

    # Now plot using the converted arrays
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label='Actual')
    plt.plot(future_years, forecast, label='Forecast')

    plt.title(f'Actual vs. Forecasted Renewable Energy Production in {country_name}')
    plt.xlabel('Year')
    plt.ylabel('Renewable Electricity (TWh)')
    plt.legend()
    plt.grid(True)
    plt.show()

def print_error_metrics(y_true, forecast):
    """
    Calculate and print error metrics for a forecast.

    Parameters:
    y_true (array-like): The true values.
    forecast (array-like): The forecasted values.

    Returns:
    None
    """
    # Calculate error metrics
    mae = mean_absolute_error(y_true, forecast)
    mse = mean_squared_error(y_true, forecast)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, forecast)

    # Print error metrics
    print('Mean Absolute Error (MAE):', mae)
    print('Mean Squared Error (MSE):', mse)
    print('Root Mean Squared Error (RMSE):', rmse)


def country_renewable_energy_forecast(country_name, merged_df, order=(1, 1, 1), forecast_steps=5):
    # Filter the DataFrame to include data only for the specified country
    country_df = merged_df[merged_df['Entity'] == country_name]

    # Convert the Pandas Series to a 1D NumPy array
    x_values = country_df['Year'].to_numpy()
    y_values = country_df['Total Renewable Electricity - TWh'].to_numpy()

    # Fit ARIMA model
    model = ARIMA(y_values, order=order)
    model_fit = model.fit()

    # Forecast for future years
    forecast = model_fit.forecast(steps=forecast_steps)

    # Generate future year indices
    future_years = np.arange(country_df['Year'].max() + 1, country_df['Year'].max() + forecast_steps + 1)

    # Print the forecasted values
    print("Forecasted renewable electricity production for", country_name + ":")
    for year, value in zip(future_years, forecast):
        print(f"Year {year}: {value} TWh")

def wind_energy_forecast(country_name, merged_df, order=(1, 1, 1), forecast_steps=5):
    # Filter the DataFrame to include data only for the specified country
    country_df = merged_df[merged_df['Entity'] == country_name]

    # Convert the Pandas Series to a 1D NumPy array
    x_values = country_df['Year'].to_numpy()
    y_values = country_df['Electricity from wind - TWh'].to_numpy()

    # Fit ARIMA model
    model = ARIMA(y_values, order=order)
    model_fit = model.fit()

    # Forecast for future years
    forecast = model_fit.forecast(steps=forecast_steps)

    # Generate future year indices
    future_years = np.arange(country_df['Year'].max() + 1, country_df['Year'].max() + forecast_steps + 1)

    # Print the forecasted values
    print("Forecasted wind electricity production for", country_name + ":")
    for year, value in zip(future_years, forecast):
        print(f"Year {year}: {value} TWh")

def solar_energy_forecast(country_name, merged_df, order=(1, 1, 1), forecast_steps=5):
    # Filter the DataFrame to include data only for the specified country
    country_df = merged_df[merged_df['Entity'] == country_name]

    # Convert the Pandas Series to a 1D NumPy array
    x_values = country_df['Year'].to_numpy()
    y_values = country_df['Electricity from solar - TWh'].to_numpy()

    # Fit ARIMA model
    model = ARIMA(y_values, order=order)
    model_fit = model.fit()

    # Forecast for future years
    forecast = model_fit.forecast(steps=forecast_steps)

    # Generate future year indices
    future_years = np.arange(country_df['Year'].max() + 1, country_df['Year'].max() + forecast_steps + 1)

    # Print the forecasted values
    print("Forecasted solar electricity production for", country_name + ":")
    for year, value in zip(future_years, forecast):
        print(f"Year {year}: {value} TWh")

def hydro_energy_forecast(country_name, merged_df, order=(1, 1, 1), forecast_steps=5):
    # Filter the DataFrame to include data only for the specified country
    country_df = merged_df[merged_df['Entity'] == country_name]

    # Convert the Pandas Series to a 1D NumPy array
    x_values = country_df['Year'].to_numpy()
    y_values = country_df['Electricity from hydro - TWh'].to_numpy()

    # Fit ARIMA model
    model = ARIMA(y_values, order=order)
    model_fit = model.fit()

    # Forecast for future years
    forecast = model_fit.forecast(steps=forecast_steps)

    # Generate future year indices
    future_years = np.arange(country_df['Year'].max() + 1, country_df['Year'].max() + forecast_steps + 1)

    # Print the forecasted values
    print("Forecasted hydro electricity production for", country_name + ":")
    for year, value in zip(future_years, forecast):
        print(f"Year {year}: {value} TWh")

def bioenergy_forecast(country_name, merged_df, order=(1, 1, 1), forecast_steps=5):
    # Filter the DataFrame to include data only for the specified country
    country_df = merged_df[merged_df['Entity'] == country_name]

    # Convert the Pandas Series to a 1D NumPy array
    x_values = country_df['Year'].to_numpy()
    y_values = country_df['Other renewables including bioenergy - TWh'].to_numpy()

    # Fit ARIMA model
    model = ARIMA(y_values, order=order)
    model_fit = model.fit()

    # Forecast for future years
    forecast = model_fit.forecast(steps=forecast_steps)

    # Generate future year indices
    future_years = np.arange(country_df['Year'].max() + 1, country_df['Year'].max() + forecast_steps + 1)

    # Print the forecasted values
    print("Forecasted bioenergy electricity production for", country_name + ":")
    for year, value in zip(future_years, forecast):
        print(f"Year {year}: {value} TWh")

def fossil_energy_forecast(country_name, merged_df, order=(1, 1, 1), forecast_steps=5):
    # Filter the DataFrame to include data only for the specified country
    country_df = merged_df[merged_df['Entity'] == country_name]

    # Convert the Pandas Series to a 1D NumPy array
    x_values = country_df['Year'].to_numpy()
    y_values = country_df['Electricity from Non-Renewables - TWh'].to_numpy()

    # Fit ARIMA model
    model = ARIMA(y_values, order=order)
    model_fit = model.fit()

    # Forecast for future years
    forecast = model_fit.forecast(steps=forecast_steps)

    # Generate future year indices
    future_years = np.arange(country_df['Year'].max() + 1, country_df['Year'].max() + forecast_steps + 1)

    # Print the forecasted values
    print("Forecasted non-renewable electricity production for", country_name + ":")
    for year, value in zip(future_years, forecast):
        print(f"Year {year}: {value} TWh")

def total_energy_forecast(country_name, merged_df, order=(1, 1, 1), forecast_steps=5):
    # Filter the DataFrame to include data only for the specified country
    country_df = merged_df[merged_df['Entity'] == country_name]

    # Convert the Pandas Series to a 1D NumPy array
    x_values = country_df['Year'].to_numpy()
    y_values = country_df['Electricity generation - TWh'].to_numpy()

    # Fit ARIMA model
    model = ARIMA(y_values, order=order)
    model_fit = model.fit()
    
    # Save the trained model to a file
    with open(f'{country_name}_total_energy_model.pkl', 'wb') as f:
        pickle.dump(model_fit, f)

    # Forecast for future years
    forecast = model_fit.forecast(steps=forecast_steps)

    # Generate future year indices
    future_years = np.arange(country_df['Year'].max() + 1, country_df['Year'].max() + forecast_steps + 1)

    # Print the forecasted values
    print("Forecasted total electricity production for", country_name + ":")
    for year, value in zip(future_years, forecast):
        print(f"Year {year}: {value} TWh")





