from pathlib import Path

import cv2

from src.dataset_loader import find_first_image
from src.features import extract_features_with_visualization
from src.preprocessing import preprocess_image


DATASET_DIR = Path("dataset")


def main():
    if not DATASET_DIR.exists():
        print("Dataset folder tidak ditemukan.")
        print("Pastikan folder dataset/ ada di project ini.")
        return

    image_path = find_first_image(DATASET_DIR)
    if image_path is None:
        print("Tidak ada image yang ditemukan di dataset.")
        return

    image = preprocess_image(image_path, debug=True)
    if image is None:
        print(f"Image tidak dapat dibaca: {image_path}")
        return

    features, hog_image = extract_features_with_visualization(image)

    print()
    print(f"Image: {image_path}")
    print(f"HOG feature shape: {features.shape}")

    cv2.imshow("Original / Grayscale Image", image)
    cv2.imshow("HOG Visualization", hog_image)
    print("Tekan tombol apa saja pada window image untuk keluar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
