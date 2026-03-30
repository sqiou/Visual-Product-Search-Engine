import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "Data/myntradataset/images")
FEATURES_PATH = os.path.join(BASE_DIR, "Data/features.pkl")
PATHS_PATH = os.path.join(BASE_DIR, "Data/paths.pkl")

IMAGE_SIZE = (224, 224)
TOP_K = 5