import cdsapi
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import data_handler

c = cdsapi.Client()

# Retrieve upper-air variable data:
c.retrieve(
    'reanalysis-era5-pressure-levels',
    {'product_type': 'reanalysis','format': 'grib','variable': ['geopotential', 'specific_humidity', 'temperature','u_component_of_wind', 'v_component_of_wind',],
        'pressure_level': ['50', '100', '150','200', '250', '300','400', '500', '600','700', '850', '925','1000',],
        'year': '2010','month': '01','day': ['01', '02', '03','04', '05', '06','07',],
        'time': ['00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',],},
    '../../weather_data/air_test.grib')
print("Air data downloaded.")

# Convert upper-air variable data into tensor:
data_handler.air_grib_to_tensor("air_test.grib", data_folder_path="../../weather_data/")
print("Air data parsed.")

# Retrieve surface variable data:
c.retrieve(
    'reanalysis-era5-single-levels',
    {'product_type': 'reanalysis','format': 'grib',
        'time': ['00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',],
        'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature','mean_sea_level_pressure',],
        'year': '2010','month': '01','day': ['01', '02', '03','04', '05', '06','07',],},
    '../../weather_data/surface_test.grib')
print("Surface data downloaded.")

# Convert surface variable data into tensor:
data_handler.surface_grib_to_tensor("surface_test.grib", data_folder_path="../../weather_data/")
print("Surface data parsed.")