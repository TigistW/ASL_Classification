from imports import *


def load_models(filename):
    with open(filename, "rb") as model_file:
        model = pickle.load(model_file)
    print(f"Model loaded successfully from {filename}.")
    return model
    