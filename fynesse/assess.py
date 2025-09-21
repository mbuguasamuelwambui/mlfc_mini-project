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
            water_bodies= combined_pois_clipped[combined_pois_clipped['natural'].isin(['water',"lake", "reservoir", "coastline", "bay", "wetland"])]
            forests     = combined_pois_clipped[(combined_pois_clipped['natural'] == 'forest')|(combined_pois_clipped['landuse'] == 'forest')]
            plantations = combined_pois_clipped[combined_pois_clipped["landuse"] == "plantation"]
            highway     = combined_pois_clipped[combined_pois_clipped["highway"].isin(tags["highway"])]
            railways    = combined_pois_clipped[combined_pois_clipped["railway"].isin(tags["railway"])]
            protected   = combined_pois_clipped[combined_pois_clipped["boundary"].isin(tags["boundary"])]
            power_lines = combined_pois_clipped[combined_pois_clipped["power"].isin(["line","cable","tower"])]
            substations = combined_pois_clipped[combined_pois_clipped["power"].isin(["substation","transformer"])]

            # Plots
            if not rivers.empty:      rivers.plot(ax=ax, color="blue", markersize=1, alpha=0.6, label="Rivers")
            if not water_bodies.empty:water_bodies.plot(ax=ax, color="cyan", markersize=5, alpha=0.6, label="Water Bodies")
            if not forests.empty:     forests.plot(ax=ax, color="darkgreen", markersize=5, alpha=0.6, label="Forests")
            if not plantations.empty: plantations.plot(ax=ax, color="limegreen", alpha=0.4, label="Plantations")
            if not highway.empty:     highway.plot(ax=ax, color="orange", linewidth=0.7, label="Roads")
            if not railways.empty:    railways.plot(ax=ax, color="purple", linewidth=1, label="Railways")
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
