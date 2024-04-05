import os
import sys
import re
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import json

# Set up data dir
dir_name = 'PGHComputerVision'
root = os.getcwd()
root = re.sub(rf"{dir_name}.*", dir_name, root)
data_root = os.path.join(root, 'Data')

def load_pgh_shp_files(EPSG_dict):

    # Define the directories containing the shapefiles
    shp_dir = os.path.join(data_root, 'mapping_shapefiles')

    # Read in PGH shapefiles
    neighbor_gdf = gpd.read_file( os.path.join(shp_dir, 'Neighborhoods_/Neighborhoods_.shp') )
    parks_gdf = gpd.read_file( os.path.join(shp_dir, 'PGHWebParks/PGHWebParks.shp') )
    roads_gdf = gpd.read_file( os.path.join(shp_dir, 'tl_2020_42003_roads/tl_2020_42003_roads.shp') )
    water_gdf = gpd.read_file( os.path.join(shp_dir, 'majrivrs/majrivrs.shp') )

    # read in json of downtown 
    with open(os.path.join(shp_dir, 'downtown_polygon.json')) as file:
        downtown_polygon = json.load(file)
    downtown_gdf = gpd.GeoDataFrame.from_features(downtown_polygon['features'])

    # read in downtown_surrounds_polygon.json and clip 4 gdfs to area
    with open(os.path.join(shp_dir, 'downtown_surroundings_polygon.json'), 'r') as file:
        downtown_surrounds_polygon = json.load(file)
    downtown_surrounds_gdf = gpd.GeoDataFrame.from_features(downtown_surrounds_polygon['features'])

    # Set CRS to WGS 84 for JSON polygons
    downtown_gdf.set_crs(epsg=4326, inplace=True)
    downtown_surrounds_gdf.set_crs(epsg=4326, inplace=True)
    
    # Transform the clipping GeoDataFrames to the target CRS
    downtown_gdf = downtown_gdf.to_crs(**EPSG_dict)
    downtown_surrounds_gdf = downtown_surrounds_gdf.to_crs(**EPSG_dict)

    # Set the CRS for each GeoDataFrame
    gdfs = [neighbor_gdf, parks_gdf, roads_gdf, water_gdf, downtown_gdf, downtown_surrounds_gdf]
    for i, gdf in enumerate(gdfs):
        gdfs[i] = gdf.to_crs(**EPSG_dict)

    # Clip neighborhood, parks, roads, and water to downtown_surrounds_polygon
    for i, gdf in enumerate(gdfs[:-2]):  # Exclude the last two GeoDataFrames (downtown and downtown_surrounds)
        gdfs[i] = gpd.clip(gdf, downtown_surrounds_gdf)

    # Return the GeoDataFrames as a tuple
    return tuple(gdfs)
