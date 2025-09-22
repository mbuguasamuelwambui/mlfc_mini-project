from typing import Any, Union
import pandas as pd
import logging

from .config import *
from . import access

# Set up logging
logger = logging.getLogger(__name__)

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded.
How are missing values encoded, how are outliers encoded? What do columns represent,
makes rure they are correctly labeled. How is the data indexed. Crete visualisation
routines to assess the data (e.g. in bokeh). Ensure that date formats are correct
and correctly timezoned."""


def data() -> Union[pd.DataFrame, Any]:
    """
    Load the data from access and ensure missing values are correctly encoded as well as
    indices correct, column names informative, date and times correctly formatted.
    Return a structured data structure such as a data frame.

    IMPLEMENTATION GUIDE FOR STUDENTS:
    ==================================

    1. REPLACE THIS FUNCTION WITH YOUR DATA ASSESSMENT CODE:
       - Load data using the access module
       - Check for missing values and handle them appropriately
       - Validate data types and formats
       - Clean and prepare data for analysis

    2. ADD ERROR HANDLING:
       - Handle cases where access.data() returns None
       - Check for data quality issues
       - Validate data structure and content

    3. ADD BASIC LOGGING:
       - Log data quality issues found
       - Log cleaning operations performed
       - Log final data summary

    4. EXAMPLE IMPLEMENTATION:
       df = access.data()
       if df is None:
           print("Error: No data available from access module")
           return None

       print(f"Assessing data quality for {len(df)} rows...")
       # Your data assessment code here
       return df
    """
    logger.info("Starting data assessment")

    # Load data from access module
    df = access.data()

    # Check if data was loaded successfully
    if df is None:
        logger.error("No data available from access module")
        print("Error: Could not load data from access module")
        return None

    logger.info(f"Assessing data quality for {len(df)} rows, {len(df.columns)} columns")

    try:
        # STUDENT IMPLEMENTATION: Add your data assessment code here

        # Example: Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"Found missing values: {missing_counts.to_dict()}")
            print(f"Missing values found: {missing_counts.sum()} total")

        # Example: Check data types
        logger.info(f"Data types: {df.dtypes.to_dict()}")

        # Example: Basic data cleaning (students should customize this)
        # Remove completely empty rows
        df_cleaned = df.dropna(how="all")
        if len(df_cleaned) < len(df):
            logger.info(f"Removed {len(df) - len(df_cleaned)} completely empty rows")

        logger.info(f"Data assessment completed. Final shape: {df_cleaned.shape}")
        return df_cleaned

    except Exception as e:
        logger.error(f"Error during data assessment: {e}")
        print(f"Error assessing data: {e}")
        return None


def query(data: Union[pd.DataFrame, Any]) -> str:
    """Request user input for some aspect of the data."""
    raise NotImplementedError


def view(data: Union[pd.DataFrame, Any]) -> None:
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError


def labelled(data: Union[pd.DataFrame, Any]) -> Union[pd.DataFrame, Any]:
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError

#clipping features form OSM to Kenya counties and power stations
import pandas as pd
import geopandas as gpd
from shapely import wkt
import matplotlib.pyplot as plt
import os, glob
import numpy as np
from sklearn.cluster import DBSCAN
from shapely.geometry import Point


def plot_features_with_clipping(kenya_counties, power_stations, csv_output_dir, tags):

    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot Kenya counties
    kenya_counties.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.5)

    # Plot power stations
    power_stations.plot(ax=ax, color="red", markersize=40, label="Power Stations")

    if os.path.exists(csv_output_dir):
        all_pois = []
        for csv_file in glob.glob(f"{csv_output_dir}/*_pois.csv"):
            try:
                pois_df = pd.read_csv(csv_file)
                all_pois.append(pois_df)
            except Exception as e:
                print(f"Could not read {csv_file}: {e}")

        if all_pois:
            combined_pois_df = pd.concat(all_pois, ignore_index=True)
            combined_pois_df['geometry'] = combined_pois_df['geometry'].apply(wkt.loads)
            combined_pois = gpd.GeoDataFrame(combined_pois_df, geometry='geometry')

            if combined_pois.crs is None:
                combined_pois = combined_pois.set_crs(kenya_counties.crs, allow_override=True)

            kenya_boundary = kenya_counties.dissolve().geometry.iloc[0]
            combined_pois_clipped = combined_pois.clip(kenya_boundary)

            # Filters
            rivers      = combined_pois_clipped[combined_pois_clipped['waterway'] == 'river']
            water_bodies= combined_pois_clipped[combined_pois_clipped['natural'].isin(['water',"lake", "reservoir"])]
            forests     = combined_pois_clipped[(combined_pois_clipped['landuse'] == 'forest')]
            protected   = combined_pois_clipped[combined_pois_clipped["boundary"].isin(tags["boundary"])]
            power_lines = combined_pois_clipped[combined_pois_clipped["power"].isin(["tower"])]
            substations = combined_pois_clipped[combined_pois_clipped["power"].isin(["substation"])]

            # Plots
            if not rivers.empty:      rivers.plot(ax=ax, color="blue", markersize=1, alpha=0.6, label="Rivers")
            if not water_bodies.empty:water_bodies.plot(ax=ax, color="cyan", markersize=5, alpha=0.6, label="Water Bodies")
            if not forests.empty:     forests.plot(ax=ax, color="darkgreen", markersize=5, alpha=0.6, label="Forests")
            if not protected.empty:   protected.plot(ax=ax, facecolor="none", edgecolor="magenta", linewidth=1, label="Protected Areas")
            if not power_lines.empty: power_lines.plot(ax=ax, color="brown", linewidth=0.8, label="Power Lines")
            if not substations.empty: substations.plot(ax=ax, color="yellow", markersize=30, label="Substations")

        else:
            print("No valid CSV files found in", csv_output_dir)
    else:
        print("CSV output directory not found:", csv_output_dir)

    plt.title("Kenya Counties, Power Stations, and OSM Features by Type", fontsize=15)
    plt.legend()
    plt.show()

def plot_county_features(county_name, kenya_counties, combined_pois_clipped, power_stations, tags):
    """
    Plots environmental features and power stations for a given county.

    Parameters:
    - county_name: str, name of the county to filter (e.g., "Nakuru")
    - kenya_counties: GeoDataFrame with county boundaries
    - combined_pois_clipped: GeoDataFrame with OSM features
    - power_stations: GeoDataFrame with power stations
    - tags: dict containing lists of values for filtering (e.g., tags["highway"], tags["railway"], tags["boundary"])
    """

    # Select county
    county = kenya_counties[kenya_counties["NAME_1"].str.contains(county_name, case=False)]

    if county.empty:
        print(f"County '{county_name}' not found.")
        return

    # Set up plot
    fig, ax = plt.subplots(figsize=(10, 10))
    county.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1)

    if combined_pois_clipped.empty:
        print("combined_pois_clipped is empty. Nothing to plot.")
        return

    # Ensure CRS is consistent
    if combined_pois_clipped.crs is None:
        combined_pois_clipped = combined_pois_clipped.set_crs(kenya_counties.crs, allow_override=True)
    else:
        combined_pois_clipped = combined_pois_clipped.to_crs(kenya_counties.crs)

    combined_pois_clipped = combined_pois_clipped.set_geometry("geometry")

    # Clip to county
    env_county = gpd.clip(combined_pois_clipped, county)

    # Separate features
    rivers       = env_county[env_county["waterway"] == "river"]
    water_bodies = env_county[env_county["natural"] == "water"]
    forests      = env_county[(env_county["natural"] == "forest") | (env_county["landuse"] == "forest")]
    protected    = env_county[env_county["boundary"].isin(tags["boundary"])]
    power_lines  = env_county[env_county["power"].isin(["tower"])]
    substations  = env_county[env_county["power"].isin(["substation"])]

    # Plot features
    if not rivers.empty: rivers.plot(ax=ax, color="blue", linewidth=1, alpha=0.7, label="Rivers")
    if not water_bodies.empty: water_bodies.plot(ax=ax, color="cyan", markersize=5, alpha=0.7, label="Water Bodies")
    if not forests.empty: forests.plot(ax=ax, color="darkgreen", markersize=5, alpha=0.7, label="Forests")
    if not protected.empty: protected.plot(ax=ax, facecolor="none", edgecolor="magenta", linewidth=1, label="Protected Areas")
    if not power_lines.empty: power_lines.plot(ax=ax, color="brown", linewidth=0.8, label="Power Lines")
    if not substations.empty: substations.plot(ax=ax, color="yellow", markersize=30, label="Substations")

    # Plot power stations in county
    power_stations_county = power_stations.clip(county.geometry.iloc[0])
    if not power_stations_county.empty:
        power_stations_county.plot(ax=ax, color="red", markersize=50, edgecolor="black", label="Power Stations")

    # Final touches
    plt.title(f"Environmental Features and Power Stations in {county_name} County", fontsize=14)
    plt.legend()
    plt.show()

    return env_county

def calculate_distances_to_features(power_stations_gdf, features_dict, target_crs=32637):
    """
    Calculates the distance from each power station to the nearest feature in multiple categories.

    Parameters:
    - power_stations_gdf: GeoDataFrame of power station locations.
    - features_dict: Dictionary of {feature_name: GeoDataFrame} where each GeoDataFrame
                     contains geometries of that feature.
    - target_crs: The projected CRS to use for distance calculations (default EPSG:32637, UTM Zone 37N).

    Returns:
    - power_stations_gdf: The input GeoDataFrame with new distance columns for each feature.
    """
    if power_stations_gdf.empty:
        print("Power stations GeoDataFrame is empty. Cannot calculate distances.")
        for feature_name in features_dict.keys():
            # Use consistent naming convention
            power_stations_gdf[f"distance_to_{feature_name}"] = None
        return power_stations_gdf

    # Reproject power stations to projected CRS
    power_stations_proj = power_stations_gdf.to_crs(target_crs)

    for feature_name, feature_gdf in features_dict.items():
        # Reproject features to projected CRS, handle empty GeoDataFrames
        if not feature_gdf.empty:
            feature_proj = feature_gdf.to_crs(target_crs)
            # Use consistent naming convention
            power_stations_gdf[f"distance_to_{feature_name}"] = power_stations_proj.geometry.apply(
                lambda geom: feature_proj.distance(geom).min()
            )
        else:
            # Use consistent naming convention
            power_stations_gdf[f"distance_to_{feature_name}"] = None

    return power_stations_gdf

def run_dbscan_clustering(gdf, eps=10000, min_samples=3, counties_gdf=None, plot=True):
    """
    Apply DBSCAN clustering to a GeoDataFrame of point geometries (e.g., power stations).

    Parameters:
    -----------
    gdf : GeoDataFrame
        GeoDataFrame containing point geometries.
    eps : float
        Maximum distance (in CRS units, e.g. meters) for neighborhood.
    min_samples : int
        Minimum number of samples to form a cluster.
    counties_gdf : GeoDataFrame, optional
        GeoDataFrame of counties for background plotting.
    plot : bool
        If True, generate a cluster visualization.

    Returns:
    --------
    gdf_out : GeoDataFrame
        Input GeoDataFrame with an added `cluster_label` column.
    cluster_counts : Series
        Count of points per cluster label.
    """
    # Ensure CRS is projected
    if gdf.crs.is_geographic:
        raise ValueError("GeoDataFrame must be in a projected CRS (meters).")

    # Get coordinates
    coords = np.vstack([gdf.geometry.x, gdf.geometry.y]).T

    # Run DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(coords)

    # Add cluster labels
    gdf_out = gdf.copy()
    gdf_out['cluster_label'] = clusters

    # Plot if requested
    if plot:
        fig, ax = plt.subplots(figsize=(12, 12))

        if counties_gdf is not None:
            counties_gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.5)

        scatter = ax.scatter(
            gdf_out.geometry.x, gdf_out.geometry.y,
            c=gdf_out['cluster_label'], cmap='viridis', s=50
        )

        plt.title("DBSCAN Clustering of Power Stations")
        plt.xlabel("Easting (m)")
        plt.ylabel("Northing (m)")
        plt.colorbar(scatter, ax=ax, label="Cluster Label")
        plt.show()

    return gdf_out, gdf_out['cluster_label'].value_counts()

def run_dbscan_with_features(gdf, eps=10000, min_samples=3):
    """
    Apply DBSCAN clustering and extract cluster-based numeric features.

    Parameters:
    -----------
    gdf : GeoDataFrame
        GeoDataFrame with point geometries (e.g., power stations).
    eps : float
        Maximum neighborhood distance (in CRS units).
    min_samples : int
        Minimum points per cluster.

    Returns:
    --------
    gdf_out : GeoDataFrame
        Copy of gdf with added columns:
        - cluster_label
        - cluster_size
        - dist_to_cluster_centroid
    """
    if gdf.crs.is_geographic:
        raise ValueError("GeoDataFrame must be projected (meters). Reproject first.")

    coords = np.vstack([gdf.geometry.x, gdf.geometry.y]).T

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(coords)

    gdf_out = gdf.copy()
    gdf_out['cluster_label'] = clusters

    # Initialize features
    gdf_out['cluster_size'] = 0
    gdf_out['dist_to_cluster_centroid'] = np.nan

    # Compute features per cluster (excluding noise = -1)
    for clust in np.unique(clusters):
        if clust == -1:  # noise points
            continue

        mask = gdf_out['cluster_label'] == clust
        cluster_points = coords[mask]

        # Cluster size
        size = len(cluster_points)
        gdf_out.loc[mask, 'cluster_size'] = size

        # Cluster centroid
        centroid_x, centroid_y = cluster_points.mean(axis=0)
        centroid_point = Point(centroid_x, centroid_y)

        # Distance to centroid
        dists = gdf_out.loc[mask].geometry.distance(centroid_point)
        gdf_out.loc[mask, 'dist_to_cluster_centroid'] = dists

    return gdf_out

def compute_power_station_density(power_stations_gdf, kenya_counties):
    # Ensure both GeoDataFrames are in the same CRS for spatial join
    power_stations_gdf = power_stations_gdf.to_crs(kenya_counties.crs)

    # Perform a spatial join to find which county each power station is in
    power_stations_with_counties = gpd.sjoin(power_stations_gdf, kenya_counties, how="inner", predicate="within")

    # Group by county name and count the number of power stations
    power_stations_per_county = power_stations_with_counties.groupby('NAME_1').size().reset_index(name='power_station_count')

    # Ensure kenya_counties is in a projected CRS for accurate area calculation
    kenya_counties_proj = kenya_counties.to_crs(epsg=32637)  # Using UTM Zone 37N

    # Project power stations to the same CRS for area calculation
    power_stations_proj = power_stations_gdf.to_crs(epsg=32637)

    # Calculate the area of each county in square kilometers
    kenya_counties_proj['area_sqkm'] = kenya_counties_proj.geometry.area / 10**6  # Convert from square meters to square kilometers

    # Merge the power station counts with the county areas
    power_stations_density = power_stations_per_county.merge(
        kenya_counties_proj[['NAME_1', 'area_sqkm']],
        on='NAME_1',
        how='left'
    )

    # Calculate density per square kilometer
    power_stations_density['density_per_sqkm'] = power_stations_density['power_station_count'] / power_stations_density['area_sqkm']

    # Display the results, sorted by density
    display(power_stations_density.sort_values(by='density_per_sqkm', ascending=False))
