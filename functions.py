import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import geopandas as gpd


# univariate analysis - countplots
def plot_top_n(df, column, n=10, title=None, smallest=False, skip_first=False):
    if skip_first:
        top_n_values = df[column].value_counts().nlargest(n + 1)[1:]
    elif smallest:
        top_n_values = df[column].value_counts().nsmallest(n)
    else:
        top_n_values = df[column].value_counts().nlargest(n)

    # Plotting
    plt.figure(figsize=(10, 6))
    top_n_values.plot(kind='bar', color='skyblue')
    plt.title(title if title else f'Top {n} {column.capitalize()}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=90, ha='right')  # Adjust rotation for better readability
    plt.show();
    

# bivariate analysis 1 vs 1
def plot_top_and_bottom_countries_by_capacity(df, n=10):
    # Calculate the top countries by average capacity in descending order
    top_countries = df.groupby('country')['capacity_mw'].mean().nlargest(n).sort_values(ascending=False).index

    # Filter the dataframe for the top countries
    df_top_countries = df[df['country'].isin(top_countries)]

    # Plotting for the top countries
    plt.figure(figsize=(12, 8))
    sns.barplot(x='country', y='capacity_mw', data=df_top_countries, ci=None, order=top_countries)
    plt.title(f'Top {n} Countries by Average Capacity (Descending Order)')
    plt.xlabel('Country')
    plt.ylabel('Average Capacity (MW)')
    plt.xticks(rotation=45)
    plt.show()

    # Calculate the bottom countries by average capacity in ascending order
    bottom_countries = df.groupby('country')['capacity_mw'].mean().nsmallest(n).sort_values(ascending=True).index

    # Filter the dataframe for the bottom countries
    df_bottom_countries = df[df['country'].isin(bottom_countries)]

    # Plotting for the bottom countries
    plt.figure(figsize=(12, 8))
    sns.barplot(x='country', y='capacity_mw', data=df_bottom_countries, ci=None, order=bottom_countries)
    plt.title(f'Bottom {n} Countries by Average Capacity (Ascending Order)')
    plt.xlabel('Country')
    plt.ylabel('Average Capacity (MW)')
    plt.xticks(rotation=45)
    plt.show();
    
    
# scatter plots    
def plot_capacity_generation_scatter(df):
    # Scatter plot for Capacity vs. Generation in 2017
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='capacity_mw', y='generation_gwh_2017', data=df)
    plt.title('Capacity vs. Generation in 2017')
    plt.xlabel('Capacity (MW)')
    plt.ylabel('Generation in 2017 (GWh)')
    plt.show()

    # Scatter plot for Capacity Variation Across Commissioning Years
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='commissioning_year', y='capacity_mw', data=df, alpha=0.5)
    plt.title('Capacity Variation Across Commissioning Years')
    plt.xlabel('Commissioning Year')
    plt.ylabel('Capacity (MW)')
    plt.xticks(rotation=45)
    plt.show()
    
    
# mapping    
def plot_power_plant_distribution(df):
    # Create a GeoDataFrame from the DataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))

    # Load a world map shapefile for context
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    # Plot the power plant locations on top of the world map
    fig, ax = plt.subplots(figsize=(12, 8))
    world.boundary.plot(ax=ax, linewidth=1)
    gdf.plot(ax=ax, markersize=10, color='red', alpha=0.7, label='Power Plants')

    # Customize the plot (add title, legend, etc.)
    plt.title('Global Power Plant Distribution')
    plt.legend()
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Show the plot
    plt.show()


    