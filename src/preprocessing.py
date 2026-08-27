from pathlib import Path

import cv2
import numpy as np


IMAGE_SIZE = (48, 48)


def preprocess_image(image, equalize_hist: bool = False, debug: bool = False) -> np.ndarray | None:
    """
    Convert image input into a clean 48x48 grayscale image.

    image can be:
    - path to an image file
    - OpenCV image array from cv2.imread or webcam crop
    """
    if isinstance(image, (str, Path)):
        image = cv2.imread(str(image))
        if image is None:
            return None

    original_shape = image.shape

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    processed = cv2.resize(gray, IMAGE_SIZE)

    if equalize_hist:
        processed = cv2.equalizeHist(processed)

    processed = processed.astype(np.uint8)

    if debug:
        print(f"Original shape: {original_shape}")
        print(f"Processed shape: {processed.shape}")
        print()
        print("Pixel example:")
        print(processed[0, :12].tolist())
        print()
        print("48 x 48 = 2304 pixels")
        print("Jika di-flatten langsung, satu image menjadi 2304 features.")

    return processed
