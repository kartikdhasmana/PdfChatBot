import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

def get_data_dir():
    return os.path.join(BASE_DIR, 'data')

def get_output_dir():
    return os.path.join(BASE_DIR, 'output')

def get_vector_store_dir():
    return os.path.join(BASE_DIR, 'vector_store')
