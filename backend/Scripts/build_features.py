import os
import pickle
import numpy as np
import sys

# allow import from parent folder
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from feature_extractor import FeatureExtractor

DATASET_PATH = "Data/images"
FEATURES_PATH = "Data/features.pkl"
PATHS_PATH = "Data/paths.pkl"

os.makedirs("Data", exist_ok=True)

BATCH_SIZE = 32
MAX_IMAGES = 1000  # start small

def build_features():
    extractor = FeatureExtractor()

    features = []
    paths = []

    all_images = []

    # collect all images
    for root, dirs, files in os.walk(DATASET_PATH):
        for file in files:
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                all_images.append(os.path.join(root, file))

    print(f"Total images found: {len(all_images)}")

    # limit for testing
    all_images = all_images[:MAX_IMAGES]

    print(f"Processing {len(all_images)} images...")

    for i, img_path in enumerate(all_images):
        try:
            feat = extractor.extract(img_path)
            feat = feat.astype("float32")

            features.append(feat)
            paths.append(img_path)

            if i % 50 == 0:
                print(f"Processed {i} images")

        except Exception as e:
            print("Error:", img_path, e)

    # save results
    with open(FEATURES_PATH, "wb") as f:
        pickle.dump(features, f)

    with open(PATHS_PATH, "wb") as f:
        pickle.dump(paths, f)

    print("Features saved successfully!")

if __name__ == "__main__":
    build_features()