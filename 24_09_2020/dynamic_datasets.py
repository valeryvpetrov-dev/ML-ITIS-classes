import requests
import json
import pandas as pd
from dateutil import parser

BASE_URL = 'http://data.kzn.ru:8082/api/v0/dynamic_datasets/'
URL_PATH_BUS = 'bus.json'


# http://data.kzn.ru/dinamic_datasets/docs
def request_bus_data_json():
    request_url = BASE_URL + URL_PATH_BUS
    response = requests.get(request_url)
    return response.json()


def save_data_to_json_file(data_json_file_path, data_json):
    with open(data_json_file_path, 'w') as outfile:
        json.dump(data_json, outfile)


def read_data_from_json_file(data_json_file_path):
    with open(data_json_file_path, 'r') as data_json_file:
        data_df = pd.json_normalize(json.load(data_json_file))
        data_df['updated_at'] = data_df['updated_at'].apply(__str_to_datetime)
        data_df['data.GaragNumb'] = data_df['data.GaragNumb'].astype(int)
        data_df['data.Graph'] = data_df['data.Graph'].astype(int)
        data_df['data.Smena'] = data_df['data.Smena'].astype(int)
        data_df['data.TimeNav'] = data_df['data.TimeNav'].apply(__str_to_datetime)
        data_df['data.Latitude'] = data_df['data.Latitude'].astype(float)
        data_df['data.Longitude'] = data_df['data.Longitude'].astype(float)
        data_df['data.Speed'] = data_df['data.Speed'].astype(int)
        data_df['data.Azimuth'] = data_df['data.Azimuth'].astype(int)
        return data_df


def __str_to_datetime(datetimeStr):
    datetimeStr = ' '.join(datetimeStr.split()[:2])
    return parser.parse(datetimeStr)


if __name__ == '__main__':
    bus_data_json_file_path = './data/bus_data.json'
    # uncomment if you need to request data from data.kzn.ru and save it to file
    # data_json = request_bus_data_json()
    # save_data_to_json_file(bus_data_json_file_path, data_json)
    data_df = read_data_from_json_file(bus_data_json_file_path)
    print(data_df.head(10))
