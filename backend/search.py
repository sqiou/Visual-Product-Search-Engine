import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import TOP_K

class ImageSearch:
    def __init__(self, features, paths):
        self.features = np.array(features)
        self.paths = paths

    def search(self, query_feature, category=None):
        similarities = cosine_similarity([query_feature], self.features)[0]

        # sort descending
        indices = similarities.argsort()[::-1]

        results = []

        for idx in indices:
            path = self.paths[idx]

            # optional category filter
            if category and category not in path:
                continue

            results.append({
                "image": path,
                "score": float(similarities[idx])
            })

            if len(results) >= TOP_K:
                break

        return results