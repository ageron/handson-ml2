import os
import tarfile
import urllib.request
import pandas as pd

housing_root = 'https://raw.githubusercontent.com/ageron/handson-ml2/master/'
housing_url = housing_root + "datasets/housing/housing.tgz"
housing_path = os.path.join('datasets', 'housing_test')

def fetch_data(housing_url = housing_url, housing_path = housing_path):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, 'housing.tgz')
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path = housing_path)
    housing_tgz.close()

def housing_data(housing_path = housing_path):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)