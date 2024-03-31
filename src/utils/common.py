import pickle

def encode(value):
    return str(value).encode()

def decode(bytes_value):
    return pickle.loads(bytes_value)
