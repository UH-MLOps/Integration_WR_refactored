import config as cfg
import warnings
warnings.filterwarnings('ignore')
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from geographiclib.geodesic import Geodesic as GD
from scipy.interpolate import interpn
from datetime import datetime, timedelta, timezone
from timezonefinder import TimezoneFinder
from dateutil import tz


def read_files():
    if cfg.FILE_NAME.endswith('.json'):
        with open(cfg.FILE_NAME) as data_file:
            data = json.load(data_file)
        df = pd.json_normalize(data, 'coordinates')
        df['timestamp'] = data['timestamps']
    else:
        df = pd.read_csv(cfg.FILE_NAME)
        df = df[list(cfg.COLUMN_NAMES.keys())]
        df = df.rename(columns = cfg.COLUMN_NAMES)
    return df


def fix_datatypes(df):
    df = convert_datetime(df)
    df = fix_longitude(df)
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs = 4326)
    return df
def convert_datetime(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
    return df.sort_values('timestamp')

def fix_longitude(df):
    df.loc[df['longitude'] < 0, 'longitude'] += 360
    return df

def filter_waypoints(df):
    filtered_df = df.cx[cfg.AREA_BOUNDING_BOX[1][1]:cfg.AREA_BOUNDING_BOX[0][1], :]
    if len(filtered_df) == 0:
        raise ValueError('Rerouting point outside the right bound of the experimenting area')
    dt = cfg.WAYPOINT_T0
    filtered_df = filtered_df[filtered_df['timestamp'] >= dt]
    index = list(filtered_df.index)
    index.extend([np.max(index) + 1, np.min(index) - 1])
    filtered_df = df.iloc[index, :]
    return filtered_df.sort_values('timestamp')

def get_bearing(lat1, lat2, long1, long2):
    '''The function is used to calculate the CoG from node 1 to node 2'''
    brng = GD.WGS84.Inverse(lat1, long1, lat2, long2)['azi1']
    if brng < 0:
        brng = brng + 360
    return brng
def calculate_waypoints(df):
    sections_waypoints = pd.DataFrame(columns=['E_latitude', 'E_longitude', 'L_latitude',
                                                    'L_longitude', 'E_timestamp', 'L_timestamp', 'cog'])
    for i in range(len(df) - 1):
        row1 = df.iloc[i]
        row2 = df.iloc[i + 1]

        # Calculate course over ground between two waypoints
        cog = get_bearing(row1['latitude'], row2['latitude'], row1['longitude'], row2['longitude'])

        # Add the information to the sections_waypoints dataframe
        sections_waypoints = sections_waypoints.append({
                'E_latitude': row1['latitude'],
                'E_longitude': row1['longitude'],
                'E_timestamp': row1['timestamp'],
                'L_latitude': row2['latitude'],
                'L_longitude': row2['longitude'],
                'L_timestamp': row2['timestamp'],
                'cog': cog
        }, ignore_index=True)
    sections_waypoints['E_latitude'] = sections_waypoints['E_latitude'].astype(float)
    sections_waypoints['E_longitude'] = sections_waypoints['E_longitude'].astype(float)
    sections_waypoints['L_latitude'] = sections_waypoints['L_latitude'].astype(float)
    sections_waypoints['L_longitude'] = sections_waypoints['L_longitude'].astype(float)
    sections_waypoints['E_timestamp'] = pd.to_datetime(sections_waypoints['E_timestamp'])
    sections_waypoints['L_timestamp'] = pd.to_datetime(sections_waypoints['L_timestamp'])
    sections_waypoints['cog'] = sections_waypoints['cog'].astype(float)
    return sections_waypoints

def find_G0(start, end):
    for val in cfg.LONG_LIST:
        if val>start and val<end:
            return val
def interpolate_points(sections_waypoints):
    interpolation_full = pd.DataFrame({'longitude': cfg.LONG_LIST})
    # Add empty columns to the DataFrame
    interpolation_full['latitude'] = np.nan  # longitudes are given according to the grids
    # latitudes are interpolated
    interpolation_full['cog'] = np.nan
    interpolation_full['timestamp'] = pd.NaT
    interpolation_full['arriving_time_latest'] = np.nan
    interpolation_full['arriving_time_previous'] = np.nan
    # return the index of the smallest longitude in grids that is larger than the current position G0[1]
    first_row = sections_waypoints[sections_waypoints.L_timestamp > cfg.WAYPOINT_T0].iloc[:1]
    WAYPOINT_G0 = find_G0(first_row.E_longitude.values[0], first_row.L_longitude.values[0])
    s_grids_lon_index = (interpolation_full['longitude'] >= WAYPOINT_G0).idxmax()
    # interpolate latitude timestamp and write cog
    interpolation_full.loc[s_grids_lon_index:, ['latitude', 'timestamp', 'cog']] = (
        interpolation_full.loc[s_grids_lon_index:, :]
        .apply(lambda x: intp_lat_ts_cog(x, sections_waypoints), axis=1))
    return interpolation_full
def intp_lat_ts_cog(row, sections_waypoints):
    """
    Finds the row in `sections_waypoints` where the logitudes in the grid `t_lon` falls between
    the `E_longitude` and `L_longitude` values. Calculates the latitude
    and timestamp for the target point T_point and sets the `latitude`,
    `timestamp`, and `cog` columns in accordingly.

    Parameters:
    t_lon (float): The longitude value to search for in `sections_waypoints`.
    sections_waypoints (pd.DataFrame): The DataFrame containing the sections and waypoints.

    Returns:
    pd.DataFrame: a row with interpolated values of latitude, timestamp and cog
    """
    # Get t_lon value
    t_lon = row['longitude']
    # Find row in `sections_waypoints` where `t_lon` falls between
    #         the `E_longitude` and `L_longitude` values
    mask = (sections_waypoints['E_longitude'] <= t_lon) & (t_lon <= sections_waypoints['L_longitude'])
    row_sw = sections_waypoints.loc[mask]
    # Calculate t_lat using linear interpolation
    point1 = np.array([row_sw['E_latitude'], row_sw['E_longitude']])
    point2 = np.array([row_sw['L_latitude'], row_sw['L_longitude']])
    t_lat = point1[0] + ((t_lon - point1[1]) / (point2[1] - point1[1])) * (point2[0] - point1[0])
    # Calculate t_timestamp using linear interpolation
    e_ts = row_sw['E_timestamp']
    l_ts = row_sw['L_timestamp']
    t_ts = e_ts + ((t_lon - point1[1])[0] / (point2[1] - point1[1])[0]) * (l_ts - e_ts)
    return pd.Series({'latitude': t_lat[0], 'timestamp': t_ts.values[0], 'cog': row_sw['cog'].iloc[0]})


def get_last_weather_forecast():
    '''return the forecast files names and the timestamps of the forecast'''
    # only keep the year month day hour minute seconds
    if cfg.WAYPOINT_T0:
        dt = cfg.WAYPOINT_T0
    else:
        tzf_obj = TimezoneFinder()
        tz_file = tz.gettz(tzf_obj.timezone_at(lng=cfg.WAYPOINT_G0[1], lat=cfg.WAYPOINT_G0[0]))
        dt = pd.Timestamp(datetime.now(tz_file))
    forecast_hour = (dt.hour // 6) * 6
    last_forecast = pytz.utc.localize(datetime(dt.year, dt.month, dt.day, forecast_hour))
    previous_forecast = last_forecast - timedelta(hours=6)
    # the most recent forecast and the previous one
    last_forecast_name = f'{last_forecast.year:02d}{last_forecast.month:02d}{last_forecast.day:02d}{last_forecast.hour:02d}'
    previous_forecast_name = f'{previous_forecast.year:02d}{previous_forecast.month:02d}{previous_forecast.day:02d}{previous_forecast.hour:02d}'
    return (last_forecast_name, previous_forecast_name), (last_forecast, previous_forecast)


def hour2timeframe(row):
    '''The timeframe is from 0 to 209, but the real hour is from 0 to 384.
    From hour 120 onwards, every three hours per timeframe.
    The function is to find out the positions in the timeframes according to the hour'''
    if row['arriving_time_latest'] > 120:
        tf_latest = (row['arriving_time_latest'] - 120) / 3 + 120
    else:
        tf_latest = row['arriving_time_latest']

    if row['arriving_time_previous'] > 120:
        tf_previous = (row['arriving_time_previous'] - 120) / 3 + 120
    else:
        tf_previous = row['arriving_time_previous']
    return pd.Series({'tf_latest': tf_latest, 'tf_previous': tf_previous})

def store_train_data(arr):
    np.save('weather_ais.npy', arr)
def interp_row(row, wf, tsn):
    '''
    The function aims to retrieve and interpolate the weather conditions at the given position according
    to its spatial coordinates and arriving time (hour)
    input:
    wf is the weather forecast array, wf_latest or wf_previous
    tsn = 'tf_latest' or 'tf_previous' according to wf
    output: weather conditions for wind, wave and swell (8)
    '''
    # Define the interpolation points as a tuple of arrays
    interp_points = (row['latitude'], row['longitude'], row[tsn], range(wf.shape[3]))

    # Interpolate the weather variable values at the interpolation points
    var_values = interpn((np.flip(cfg.LAT_LIST), cfg.LONG_LIST, range(wf.shape[2]), range(wf.shape[3])),
                         wf, interp_points, method='linear')
    return var_values
def interpolate_corresponding_weather_attributes(interpolation_full):
    # Define a function to calculate the time difference in hours
    # between the arriving time at a location and the starting time of the weather forecast
    s_grids_lon_index = (interpolation_full['longitude'] >= cfg.WAYPOINT_G0[1]).idxmax()
    forecast_name, forecast_ts = get_last_weather_forecast()
    # read in the needed weather forecast file
    wf_latest = np.load('gfs_NP_' + forecast_name[0] + '.npy')
    wf_previous = np.load('gfs_NP_' + forecast_name[1] + '.npy')
    # flip for the 1st dimension (latitude) because interpolation requires first dimension in accending order
    # slice and drop the last longitude as weather attributes are not needed for the destination
    wf_latest = np.flip(wf_latest, axis=0)[:, :-1, :, :]
    wf_previous = np.flip(wf_previous, axis=0)[:, :-1, :, :]
    time_diff0 = lambda x: (pytz.utc.localize(x['timestamp']) - forecast_ts[0]) / timedelta(hours=1)
    time_diff1 = lambda x: (pytz.utc.localize(x['timestamp']) - forecast_ts[1]) / timedelta(hours=1)

    # Apply the function to the dataframe
    interpolation_full.loc[s_grids_lon_index:, 'arriving_time_latest'] = (interpolation_full.loc[s_grids_lon_index:, :]
                                                                          .apply(time_diff0, axis=1))
    interpolation_full.loc[s_grids_lon_index:, 'arriving_time_previous'] = (
        interpolation_full.loc[s_grids_lon_index:, :]
        .apply(time_diff1, axis=1))
    interpolation_full.loc[s_grids_lon_index:, ['tf_latest', 'tf_previous']] = (
        interpolation_full.loc[s_grids_lon_index:, :]
        .apply(hour2timeframe, axis=1))
    # interpolation of weather conditions forecasted in the latest weather forecast and the previous weather forecast
    interpolation_full.loc[s_grids_lon_index:, 'latest_weather'] = interpolation_full.loc[s_grids_lon_index:, :].apply(
        lambda row:
        interp_row(row, wf_latest, 'tf_latest'), axis=1)
    interpolation_full.loc[s_grids_lon_index:, 'previous_weather'] = interpolation_full.loc[s_grids_lon_index:,
                                                                     :].apply(lambda row:
                                                                              interp_row(row, wf_previous,
                                                                                         'tf_previous'), axis=1)
    converted_weather = interpolation_full.loc[s_grids_lon_index:, :].apply(lambda row: convert_weather_array(
        row['latest_weather'], row['cog']) - convert_weather_array(row['previous_weather'], row['cog']), axis=1)

    # Concatenate the converted weather arrays into a single flattened array
    flattened_weather = np.concatenate(converted_weather.to_numpy())

    # Zero-pad the flattened array if necessary
    if len(flattened_weather) < 59 * 8:
        padding = np.zeros((59 * 8 - len(flattened_weather),))
        flattened_weather = np.concatenate((padding, flattened_weather))
    return flattened_weather
def convert_weather_array(weather_array, cog):
    # Define a function that converts a 1D numpy array of weather variables to the desired format
    # Make a copy of the input array of weather conditions
    converted_array = np.copy(weather_array)

    # Define the indices of the directions of wind, wave and swell
    direction_indices = [1, 4, 7]

    # Subtract the cog from the direction and take the absolute value and convert if larger than 180 degree
    for idx in direction_indices:
        diff = abs(weather_array[idx] - cog)
        if diff <= 180:
            converted_array[idx] = diff
        else:
            converted_array[idx] = 360 - diff
    return converted_array

if __name__ == '__main__':
    df = read_files()
    df = fix_datatypes(df)
    df = filter_waypoints(df)
    waypoints = calculate_waypoints(df)
    waypoints = interpolate_points(waypoints)
    weather_points = interpolate_corresponding_weather_attributes(waypoints)
    store_train_data(weather_points)