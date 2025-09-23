"""
Access module for the fynesse framework.

This module handles data access functionality including:
- Data loading from various sources (web, local files, databases)
- Legal compliance (intellectual property, privacy rights)
- Ethical considerations for data usage
- Error handling for access issues

Legal and ethical considerations are paramount in data access.
Ensure compliance with e.g. .GDPR, intellectual property laws, and ethical guidelines.

Best Practice on Implementation
===============================

1. BASIC ERROR HANDLING:
   - Use try/except blocks to catch common errors
   - Provide helpful error messages for debugging
   - Log important events for troubleshooting

2. WHERE TO ADD ERROR HANDLING:
   - File not found errors when loading data
   - Network errors when downloading from web
   - Permission errors when accessing files
   - Data format errors when parsing files

3. SIMPLE LOGGING:
   - Use print() statements for basic logging
   - Log when operations start and complete
   - Log errors with context information
   - Log data summary information

4. EXAMPLE PATTERNS:
   
   Basic error handling:
   try:
       df = pd.read_csv('data.csv')
   except FileNotFoundError:
       print("Error: Could not find data.csv file")
       return None
   
   With logging:
   print("Loading data from data.csv...")
   try:
       df = pd.read_csv('data.csv')
       print(f"Successfully loaded {len(df)} rows of data")
       return df
   except FileNotFoundError:
       print("Error: Could not find data.csv file")
       return None
"""

from typing import Any, Union
import pandas as pd
import logging
import osmnx as ox
import geopandas as gpd
import contextily as cx
import matplotlib.pyplot as plt
import os, shutil

from google.colab import drive

# Set up basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def data() -> Union[pd.DataFrame, None]:
    """
    Read the data from the web or local file, returning structured format such as a data frame.

    IMPLEMENTATION GUIDE
    ====================

    1. REPLACE THIS FUNCTION WITH YOUR ACTUAL DATA LOADING CODE:
       - Load data from your specific sources
       - Handle common errors (file not found, network issues)
       - Validate that data loaded correctly
       - Return the data in a useful format

    2. ADD ERROR HANDLING:
       - Use try/except blocks for file operations
       - Check if data is empty or corrupted
       - Provide helpful error messages

    3. ADD BASIC LOGGING:
       - Log when you start loading data
       - Log success with data summary
       - Log errors with context

    4. EXAMPLE IMPLEMENTATION:
       try:
           print("Loading data from data.csv...")
           df = pd.read_csv('data.csv')
           print(f"Successfully loaded {len(df)} rows, {len(df.columns)} columns")
           return df
       except FileNotFoundError:
           print("Error: data.csv file not found")
           return None
       except Exception as e:
           print(f"Error loading data: {e}")
           return None

    Returns:
        DataFrame or other structured data format
    """
    logger.info("Starting data access operation")

    try:
        # IMPLEMENTATION: Replace this with your actual data loading code
        # Example: Load data from a CSV file
        logger.info("Loading data from data.csv")
        df = pd.read_csv("data.csv")

        # Basic validation
        if df.empty:
            logger.warning("Loaded data is empty")
            return None

        logger.info(
            f"Successfully loaded data: {len(df)} rows, {len(df.columns)} columns"
        )
        return df

    except FileNotFoundError:
        logger.error("Data file not found: data.csv")
        print("Error: Could not find data.csv file. Please check the file path.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        print(f"Error loading data: {e}")
        return None

#Downloading datasets
def fetch_and_store_dataset(url, dest_folder="/content/drive/MyDrive/mlfc_miniproject/final_mlfc"):
    # Mount Google Drive once
    drive.mount('/content/drive', force_remount=True)
    
    # Download & unzip
    zip_name = url.split("/")[-1]
    os.system(f"wget -q {url} -O {zip_name}")
    os.system(f"unzip -q -o {zip_name} -d /content/tmp_dataset")

    # Move to Drive
    os.makedirs(dest_folder, exist_ok=True)
    for item in os.listdir("/content/tmp_dataset"):
        src = os.path.join("/content/tmp_dataset", item)
        dst = os.path.join(dest_folder, item)
        if os.path.exists(dst): shutil.rmtree(dst) if os.path.isdir(dst) else os.remove(dst)
        shutil.move(src, dst)

    print(f"✅ Dataset stored in {dest_folder}")


#Querying the OSM features
tags = {
    "waterway": ["river", "stream", "canal", "dam"],
    "natural": ["water", "lake", "reservoir", "coastline", "bay", "wetland"],
    "landuse": ["forest", "farmland", "grass", "plantation", "basin"],
    "power": ["tower", "substation","generator", "transformer"],
    "highway": [ "primary", "secondary"],
    "railway": ["rail"],
    "boundary": ["protected_area", "national_park"]
}


def plot_osm_features_by_county(kenya_counties, tags, save=False, output_dir="/content/drive/MyDrive/mlfc_miniproject/final_mlfc/county_maps", save_csv=False, csv_output_dir="/content/drive/MyDrive/mlfc_miniproject/final_mlfc/county_pois_csv"):
    """
    Loops through each county in Kenya, queries OSM features using tags,
    and plots them individually. Optionally saves plots and queried data to CSV.

    Parameters:
    - kenya_counties: GeoDataFrame of Kenya counties
    - tags: Dictionary of OSM tags to query
    - save: If True, saves each plot as PNG
    - output_dir: Directory to save plots
    - save_csv: If True, saves queried features to CSV
    - csv_output_dir: Directory to save CSV files
    """

    if save and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if save_csv and not os.path.exists(csv_output_dir):
        os.makedirs(csv_output_dir)


    for _, county in kenya_counties.iterrows():
        name = county["NAME_1"]

        north,west,south, east = county.geometry.bounds
        bbox = (north,west,south, east)
        print("bbox",bbox)

        try:
            pois = ox.features_from_bbox(bbox, tags)
        except Exception as e:
            print(f"Skipping {name} due to error: {e}")
            continue

        if save_csv and not pois.empty:
            # Ensure pois is a GeoDataFrame before saving to CSV
            if isinstance(pois, gpd.GeoDataFrame):
                pois.to_csv(f"{csv_output_dir}/{name.replace(' ', '_')}_pois.csv")
            else:
                print(f"Skipping CSV save for {name}: queried features are not a GeoDataFrame.")


        fig, ax = plt.subplots(figsize=(10, 10))
        gpd.GeoSeries(county.geometry).plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1)
        if not pois.empty:
            pois.plot(ax=ax, color="green", markersize=5, alpha=0.6)

        ax.set_title(f"{name}: Environmental & Power Features", fontsize=14)
        ax.set_xlim(west, east)
        ax.set_ylim(south,north)

        # add basemap
        try:
            cx.add_basemap(ax, crs=kenya_counties.crs.to_string(), source=cx.providers.OpenStreetMap.Mapnik)
        except Exception:
            pass


        if save:
            plt.savefig(f"{output_dir}/{name.replace(' ', '_')}.png", dpi=300)
        else:
            plt.show()

        plt.close()


    

