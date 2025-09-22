"""
Address module for the fynesse framework.

This module handles question addressing functionality including:
- Statistical analysis
- Predictive modeling
- Data visualization for decision-making
- Dashboard creation
"""

from typing import Any, Union
import pandas as pd
import logging
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

# Set up logging
logger = logging.getLogger(__name__)

# Here are some of the imports we might expect
# import sklearn.model_selection  as ms
# import sklearn.linear_model as lm
# import sklearn.svm as svm
# import sklearn.naive_bayes as naive_bayes
# import sklearn.tree as tree

# import GPy
# import torch
# import tensorflow as tf

# Or if it's a statistical analysis
# import scipy.stats


def analyze_data(data: Union[pd.DataFrame, Any]) -> dict[str, Any]:
    """
    Address a particular question that arises from the data.

    IMPLEMENTATION GUIDE FOR STUDENTS:
    ==================================

    1. REPLACE THIS FUNCTION WITH YOUR ANALYSIS CODE:
       - Perform statistical analysis on the data
       - Create visualizations to explore patterns
       - Build models to answer specific questions
       - Generate insights and recommendations

    2. ADD ERROR HANDLING:
       - Check if input data is valid and sufficient
       - Handle analysis failures gracefully
       - Validate analysis results

    3. ADD BASIC LOGGING:
       - Log analysis steps and progress
       - Log key findings and insights
       - Log any issues encountered

    4. EXAMPLE IMPLEMENTATION:
       if data is None or len(data) == 0:
           print("Error: No data available for analysis")
           return {}

       print("Starting data analysis...")
       # Your analysis code here
       results = {"sample_size": len(data), "analysis_complete": True}
       return results
    """
    logger.info("Starting data analysis")

    # Validate input data
    if data is None:
        logger.error("No data provided for analysis")
        print("Error: No data available for analysis")
        return {"error": "No data provided"}

    if len(data) == 0:
        logger.error("Empty dataset provided for analysis")
        print("Error: Empty dataset provided for analysis")
        return {"error": "Empty dataset"}

    logger.info(f"Analyzing data with {len(data)} rows, {len(data.columns)} columns")

    try:
        # STUDENT IMPLEMENTATION: Add your analysis code here

        # Example: Basic data summary
        results = {
            "sample_size": len(data),
            "columns": list(data.columns),
            "data_types": data.dtypes.to_dict(),
            "missing_values": data.isnull().sum().to_dict(),
            "analysis_complete": True,
        }

        # Example: Basic statistics (students should customize this)
        numeric_columns = data.select_dtypes(include=["number"]).columns
        if len(numeric_columns) > 0:
            results["numeric_summary"] = data[numeric_columns].describe().to_dict()

        logger.info("Data analysis completed successfully")
        print(f"Analysis completed. Sample size: {len(data)}")

        return results

    except Exception as e:
        logger.error(f"Error during data analysis: {e}")
        print(f"Error analyzing data: {e}")
        return {"error": str(e)}

def generate_grid(boundary_gdf, spacing_km=10):
    """
    Generates a grid of points over Kenya with specified spacing in kilometers.
    """
    # Reproject to UTM for accurate spacing
    boundary_proj = boundary_gdf.to_crs(epsg=32637)
    bounds = boundary_proj.total_bounds

    # Create grid coordinates
    x_coords = np.arange(bounds[0], bounds[2], spacing_km * 1000)
    y_coords = np.arange(bounds[1], bounds[3], spacing_km * 1000)
    grid_points = [Point(x, y) for x in x_coords for y in y_coords]

    # Create GeoDataFrame
    grid_gdf = gpd.GeoDataFrame(geometry=grid_points, crs=boundary_proj.crs)


def label_grid_points(grid_gdf, power_stations_gdf, threshold_m=50000):
    """
    Adds a binary column 'has_power_station' to grid_gdf based on proximity to power stations.
    """
    # Reproject both to UTM
    grid_proj = grid_gdf.to_crs(epsg=32637)
    stations_proj = power_stations_gdf.to_crs(epsg=32637)

    # Check proximity
    grid_proj['has_power_station'] = grid_proj.geometry.apply(
        lambda pt: stations_proj.distance(pt).min() <= threshold_m
    ).astype(int)
    return grid_proj

    # Clip to Kenya boundary
    kenya_shape = boundary_proj.geometry.union_all()
    grid_gdf = grid_gdf[grid_gdf.geometry.within(kenya_shape)]

    return grid_gdf.to_crs(epsg=4326)

def add_environmental_features(grid_gdf, rivers_gdf, water_bodies_gdf, forests_gdf, power_stations_gdf, protected_areas_gdf):
    grid_proj = grid_gdf.to_crs(epsg=32637)
    rivers_proj = rivers_gdf.to_crs(epsg=32637)
    water_proj = water_bodies_gdf.to_crs(epsg=32637)
    forest_proj = forests_gdf.to_crs(epsg=32637)
    stations_proj = power_stations_gdf.to_crs(epsg=32637) # Reproject power stations
    protected_proj = protected_areas_gdf.to_crs(epsg=32637) # Reproject protected areas


    grid_proj['distance_to_river'] = grid_proj.geometry.apply(
        lambda geom: rivers_proj.distance(geom).min() if not rivers_proj.empty else np.nan
    )
    grid_proj['distance_to_water'] = grid_proj.geometry.apply(
        lambda geom: water_proj.distance(geom).min() if not water_proj.empty else np.nan
    )
    grid_proj['distance_to_forest'] = grid_proj.geometry.apply(
        lambda geom: forest_proj.distance(geom).min() if not forest_proj.empty else np.nan
    )
    grid_proj['distance_to_substation'] = grid_proj.geometry.apply(
        lambda geom: stations_proj.distance(geom).min() if not stations_proj.empty else np.nan
    )
    grid_proj['distance_to_protected'] = grid_proj.geometry.apply(
        lambda geom: protected_proj.distance(geom).min() if not protected_proj.empty else np.nan
    )
    return grid_proj

def prepare_gp_data(grid_gdf):
    X = grid_gdf.drop(columns=['geometry', 'has_power_station']).values
    y = grid_gdf['has_power_station'].values
    return X, y
def prepare_gp_training_data(kenya_counties, power_stations_gdf, features, spacing_km=10, threshold_m=50000):
    """
    Prepares grid-based training data for a Gaussian Process model using Kenya's spatial and environmental features.

    Parameters:
    - kenya_counties: GeoDataFrame of Kenyan counties
    - power_stations_gdf: GeoDataFrame of power stations
    - features: Dictionary of environmental GeoDataFrames (river, water_body, forest, protected)
    - spacing_km: Grid spacing in kilometers
    - threshold_m: Proximity threshold in meters for labeling grid points

    Returns:
    - X: Feature matrix (NumPy array)
    - y: Target labels (NumPy array)
    - grid_with_features: GeoDataFrame with enriched grid points
    """
    if kenya_counties is None or power_stations_gdf is None or features is None:
        print("Cannot prepare data: Required input variables are not available.")
        return None, None, None

    try:
        kenya_boundary = kenya_counties.dissolve()
        grid = generate_grid(kenya_boundary, spacing_km=spacing_km)
        grid_labeled = label_grid_points(grid, power_stations_gdf, threshold_m=threshold_m)

        # Safely extract environmental layers from features dictionary
        rivers_gdf = features.get("river", gpd.GeoDataFrame(geometry=[], crs=kenya_counties.crs))
        water_bodies_gdf = features.get("water_body", gpd.GeoDataFrame(geometry=[], crs=kenya_counties.crs))
        forests_gdf = features.get("forest", gpd.GeoDataFrame(geometry=[], crs=kenya_counties.crs))
        protected_areas_gdf = features.get("protected", gpd.GeoDataFrame(geometry=[], crs=kenya_counties.crs))

        grid_with_features = add_environmental_features(
            grid_labeled,
            rivers_gdf,
            water_bodies_gdf,
            forests_gdf,
            power_stations_gdf,
            protected_areas_gdf
        )

        X, y = prepare_gp_data(grid_with_features)
        return X, y, grid_with_features

    except Exception as e:
        print(f"Error during data preparation: {e}")
        return None, None, None

def train_gp_classifier(grid_with_features):
    """
    Trains a Gaussian Process Classifier on grid-based environmental features.

    Parameters:
    - grid_with_features: GeoDataFrame containing environmental features and 'has_power_station' label

    Returns:
    - gpc: Trained GaussianProcessClassifier model (or None if training fails)
    """
    # Extract features and target
    try:
        X = grid_with_features.drop(columns=['geometry', 'has_power_station']).values
        y = grid_with_features['has_power_station'].values
    except Exception as e:
        print(f"Error extracting features and labels: {e}")
        return None

    # Handle NaNs by replacing with column means
    if X.size > 0:
        col_means = np.nanmean(X, axis=0)
        col_means[np.isnan(col_means)] = 0  # Replace all-NaN columns with 0
        nan_indices = np.isnan(X)
        X[nan_indices] = np.take(col_means, np.where(nan_indices)[1])
    else:
        print("Feature matrix X is empty. Cannot train the model.")
        return None

    # Define and train the Gaussian Process Classifier
    kernel = 1.0 * RBF(1.0)
    gpc = GaussianProcessClassifier(kernel=kernel, random_state=0, n_jobs=-1)

    print("Training the Gaussian Process Classifier...")
    if X.size > 0 and y.size > 0 and X.shape[0] == y.shape[0]:
        try:
            gpc.fit(X, y)
            print("Model training complete.")
            return gpc
        except Exception as e:
            print(f"Error during model training: {e}")
            return None
    else:
        print("Cannot train the model: Feature matrix or target variable is empty or shape mismatch.")
        return None
def predict_power_station_probability(gpc, X, grid_with_features):
    """
    Predicts the probability of power station presence across a spatial grid using a trained GPC model.

    Parameters:
    - gpc: Trained GaussianProcessClassifier model
    - X: Feature matrix (NumPy array)
    - grid_with_features: GeoDataFrame containing grid points and features

    Returns:
    - grid_with_features: Updated GeoDataFrame with 'power_station_probability' column
    """
    if gpc is not None and X is not None and X.size > 0:
        print("Predicting probabilities...")
        try:
            y_prob = gpc.predict_proba(X)[:, 1]
            grid_with_features['power_station_probability'] = y_prob
            print("Prediction complete.")
            display(grid_with_features[['has_power_station', 'power_station_probability']].head())
        except Exception as e:
            print(f"Error during prediction: {e}")
            grid_with_features['power_station_probability'] = None
    else:

import matplotlib.pyplot as plt

def visualize_power_station_probability(grid_with_features, power_stations_proj=None):
    """
    Visualizes the predicted probability of power station presence across Kenya.

    Parameters:
    - grid_with_features: GeoDataFrame with 'power_station_probability' column
    - power_stations_proj: (Optional) GeoDataFrame of actual power station locations in projected CRS
    """
    if 'power_station_probability' in grid_with_features.columns and not grid_with_features.empty:
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))

        # Plot the probability heatmap
        grid_with_features.plot(
            column='power_station_probability',
            ax=ax,
            legend=True,
            cmap='OrRd',
            legend_kwds={
                'label': "Probability of Power Station Presence",
                'orientation': "horizontal"
            }
        )

        # Overlay actual power station locations if provided
        if power_stations_proj is not None and not power_stations_proj.empty:
            power_stations_proj.plot(
                ax=ax,
                color='blue',
                markersize=10,
                label='Actual Power Stations'
            )

        plt.title('Predicted Probability of Power Station Presence in Kenya')
        plt.xlabel('Easting (meters)')
        plt.ylabel('Northing (meters)')
        plt.legend()
        plt.show()
    else:
        print("Predicted probability column not found or grid data is empty. Cannot visualize.")

        print("Model not trained or feature matrix not available. Cannot predict probabilities.")
        grid_with_features['power_station_probability'] = None

    return grid_with_features
