from pathlib import Path

from src.dataset_loader import EMOTION_CLASSES, count_images, find_first_image
from src.preprocessing import preprocess_image


DATASET_DIR = Path("dataset")


def main():
    if not DATASET_DIR.exists():
        print("Dataset folder tidak ditemukan.")
        print("Pastikan folder dataset/ ada di project ini.")
        return

    counts = count_images(DATASET_DIR)
    total_images = 0

    print("FER-2013 Dataset")
    print()

    for split_name, split_counts in counts.items():
        print(f"[{split_name}]")
        for class_name in EMOTION_CLASSES:
            image_count = split_counts[class_name]
            total_images += image_count
            print(f"{class_name:10s}: {image_count}")
        print()

    print(f"Total Images: {total_images}")
    print()

    example_path = find_first_image(DATASET_DIR)
    if example_path is None:
        print("Tidak ada image yang ditemukan di dataset.")
        return

    image = preprocess_image(example_path)
    if image is None:
        print(f"Image contoh tidak dapat dibaca: {example_path}")
        return

    print(f"Example image: {example_path}")
    print(f"Example image shape: {image.shape}")
    print(f"dtype: {image.dtype}")
    print(f"min pixel: {image.min()}")
    print(f"max pixel: {image.max()}")
    print()
    print("Image adalah matrix angka. Untuk FER-2013:")
    print("48 x 48 = 2304 pixels")


if __name__ == "__main__":
    main()
