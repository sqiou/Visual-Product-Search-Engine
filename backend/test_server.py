from feature_extractor import FeatureExtractor
from search import ImageSearch
from utils import load_pickle

FEATURES_PATH = "Data/features.pkl"
PATHS_PATH = "Data/paths.pkl"

# Load data
features = load_pickle(FEATURES_PATH)
paths = load_pickle(PATHS_PATH)

# Initialize extractor
extractor = FeatureExtractor()

# Initialize search
search_engine = ImageSearch(features, paths)

# Test image (PUT ANY IMAGE PATH HERE)
query_image = "Data/images/1164.jpg"

query_feat = extractor.extract(query_image)

results = search_engine.search(query_feat)

for r in results:
    print(r)