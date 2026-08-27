import cv2
import numpy as np
from skimage.feature import hog


HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
    "feature_vector": True,
}


def extract_features(image: np.ndarray, debug: bool = False) -> np.ndarray:
    """
    Extract HOG features from a 48x48 grayscale image.

    For 48x48 with 8x8 cells:
    - 6 cells per row and column
    - 5x5 HOG blocks
    - 5 * 5 * 2 * 2 * 9 = 900 features
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_float = image.astype(np.float32) / 255.0
    features = hog(image_float, **HOG_PARAMS)

    if debug:
        print("Image shape:")
        print(image.shape)
        print()
        print("HOG feature shape:")
        print(features.shape)
        print()
        print("First 10 HOG features:")
        print(np.round(features[:10], 4).tolist())

    return features


def extract_features_with_visualization(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image_float = image.astype(np.float32) / 255.0
    features, hog_image = hog(image_float, visualize=True, **HOG_PARAMS)
    hog_image = (hog_image / hog_image.max() * 255).astype(np.uint8) if hog_image.max() > 0 else hog_image
    return features, hog_image
