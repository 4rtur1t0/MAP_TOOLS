import numpy as np
import pyproj
from pyproj import Proj
import pandas as pd


def gps2utm(df_gps, config_ref):
    """
    Projects lat, lon to UTM coordinates
    using the origin (first lat, lon)
    """
    latitude = df_gps['latitude']
    longitude = df_gps['longitude']
    altitude = df_gps['altitude']
    # status = df_gps['status']

    # base reference system
    lat_ref = config_ref['latitude']
    lon_ref = config_ref['longitude']
    altitude_ref = config_ref['altitude']

    # status_array = np.array(status)
    myProj = Proj(proj='utm', zone='30', ellps='WGS84', datum='WGS84', preserve_units=False,
                  units='m')

    lat = np.array(latitude)
    lon = np.array(longitude)
    altitude = np.array(altitude)

    UTMx_ref, UTMy_ref = myProj(lon_ref, lat_ref)
    UTMx, UTMy = myProj(lon, lat)
    # UTMx = UTMx[idx]
    # UTMy = UTMy[idx]
    UTMx = UTMx - UTMx_ref
    UTMy = UTMy - UTMy_ref
    altitude = altitude - altitude_ref
    # df_gps.insert(2, 'x', UTMx, True)
    # df_gps.insert(2, 'y', UTMy, True)
    df_gps['x'] = UTMx
    df_gps['y'] = UTMy
    df_gps['altitude'] = altitude
    return df_gps


def utm2gps(utm_x, utm_y):
    """
    Projects lat, lon to UTM coordinates
    using the origin (first lat, lon)
    """
    utm_zone = 30  # UTM Zone
    utm_letter = 'T'  # UTM Latitude Band

    # base reference system
    # lat_ref = config_ref['latitude']
    # lon_ref = config_ref['longitude']
    # altitude_ref = config_ref['altitude']

    utm_proj = pyproj.CRS(f"EPSG:326{utm_zone}")

    # Create the WGS84 (GPS) projection
    wgs84_proj = pyproj.CRS("EPSG:4326")  # WGS84 projection for GPS (latitude and longitude)

    # Transform the UTM coordinates to WGS84 (latitude, longitude)
    transformer = pyproj.Transformer.from_crs(utm_proj, wgs84_proj, always_xy=True)
    longitude, latitude = transformer.transform(utm_x, utm_y)

    # # status_array = np.array(status)
    # myProj = Proj(proj='utm', zone='30', ellps='WGS84', datum='WGS84', preserve_units=False,
    #               units='m')
    #
    # lat = np.array(latitude)
    # lon = np.array(longitude)
    # altitude = np.array(altitude)
    #
    # UTMx_ref, UTMy_ref = myProj(lon_ref, lat_ref)
    # UTMx, UTMy = myProj(lon, lat)
    # # UTMx = UTMx[idx]
    # # UTMy = UTMy[idx]
    # UTMx = UTMx - UTMx_ref
    # UTMy = UTMy - UTMy_ref
    # altitude = altitude - altitude_ref
    # # df_gps.insert(2, 'x', UTMx, True)
    # # df_gps.insert(2, 'y', UTMy, True)
    # df_gps['x'] = UTMx
    # df_gps['y'] = UTMy
    # df_gps['altitude'] = altitude
    return latitude, longitude


def filter_gps(df_gps):
    """
    Filters NaN and 0.0 in data.
    """
    # df_copy = df_gps.iloc[:0, :].copy()
    data = []
    for i in range(len(df_gps)):
        if df_gps['latitude'][i] != 0.0:
            data.append(df_gps.iloc[i])
    df_gps_out = pd.DataFrame(data)
    # df_gps = df_gps.dropna()
    # Looking for zeroes in latitude
    # df_gps = df_gps.loc[abs(df_gps['latitude']) > 0.0]
    # ~(df_gps['altitude'] == 0.0)
    # df_gps = df_gps[~(df_gps['altitude'] == 0.0)]
    # df_gps = df_gps.loc[(df_gps['latitude'] == 0.0).any(axis=0)]
    # df_gps = df_gps[~(df_gps['latitude'] == 0.0).values]
    return df_gps_out