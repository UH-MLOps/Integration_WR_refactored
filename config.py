import numpy as np


FILE_NAME = 'examples/error_voyage.json'
COLUMN_NAMES = {'TIME AT WAYPOINT': 'timestamp', 'LATITUDE': 'latitude', 'LONGITUDE': 'longitude'}
AREA_BOUNDING_BOX = [(50, 156.00000694), (25, 228.50001016)] # upper left, lower right corner

MODEL_NAME = 'pretrained_wr_v2.pt'
MODEL_INPUT = 'weather_ais.npy'
LONG_LIST = np.array([156.00000694, 157.25000699, 158.50000705, 159.7500071 ,
                      161.00000716, 162.25000722, 163.50000727, 164.75000733,
                      166.00000738, 167.25000744, 168.50000749, 169.75000755,
                      171.00000761, 172.25000766, 173.50000772, 174.75000777,
                      176.00000783, 177.25000788, 178.50000794, 179.75000799,
                      181.00000805, 182.25000811, 183.50000816, 184.75000822,
                      186.00000827, 187.25000833, 188.50000838, 189.75000844,
                      191.00000849, 192.25000855, 193.50000861, 194.75000866,
                      196.00000872, 197.25000877, 198.50000883, 199.75000888,
                      201.00000894, 202.250009  , 203.50000905, 204.75000911,
                      206.00000916, 207.25000922, 208.50000927, 209.75000933,
                      211.00000938, 212.25000944, 213.5000095 , 214.75000955,
                      216.00000961, 217.25000966, 218.50000972, 219.75000977,
                      221.00000983, 222.25000988, 223.50000994, 224.75001000,
                      226.00001005, 227.25001011, 228.50001016])
LAT_LIST = np.array([50., 49.75, 49.5 , 49.25, 49., 48.75, 48.5, 48.25, 48.,
       47.75, 47.5 , 47.25, 47.  , 46.75, 46.5 , 46.25, 46.  , 45.75,
       45.5 , 45.25, 45.  , 44.75, 44.5 , 44.25, 44.  , 43.75, 43.5 ,
       43.25, 43.  , 42.75, 42.5 , 42.25, 42.  , 41.75, 41.5 , 41.25,
       41.  , 40.75, 40.5 , 40.25, 40.  , 39.75, 39.5 , 39.25, 39.  ,
       38.75, 38.5 , 38.25, 38.  , 37.75, 37.5 , 37.25, 37.  , 36.75,
       36.5 , 36.25, 36.  , 35.75, 35.5 , 35.25, 35.  , 34.75, 34.5 ,
       34.25, 34.  , 33.75, 33.5 , 33.25, 33.  , 32.75, 32.5 , 32.25,
       32.  , 31.75, 31.5 , 31.25, 31.  , 30.75, 30.5 , 30.25, 30.  ,
       29.75, 29.5 , 29.25, 29.  , 28.75, 28.5 , 28.25, 28.  , 27.75,
       27.5 , 27.25, 27.  , 26.75, 26.5 , 26.25, 26.  , 25.75, 25.5 ,
       25.25])

USE_MINIO = False

MINIO_HOST = 'datalake.vessel-ai.eu'
MINIO_ACCESS_KEY = 'vesselaiminio'
MINIO_SECRET_KEY = 'vesselaiminio123'
MINIO_BUCKET_NAME = 'pilot4-weather-data'