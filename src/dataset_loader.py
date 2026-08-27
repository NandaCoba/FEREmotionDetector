from pathlib import Path

import numpy as np

from src.features import extract_features
from src.preprocessing import preprocess_image


EMOTION_CLASSES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def count_images(dataset_dir: str | Path = "dataset") -> dict[str, dict[str, int]]:
    dataset_dir = Path(dataset_dir)
    counts: dict[str, dict[str, int]] = {}

    for split_dir in get_split_dirs(dataset_dir):
        split_name = split_dir.name
        counts[split_name] = {}
        for class_name in EMOTION_CLASSES:
            class_dir = split_dir / class_name
            counts[split_name][class_name] = len(list_image_paths(class_dir)) if class_dir.exists() else 0

    return counts


def get_split_dirs(dataset_dir: Path) -> list[Path]:
    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"

    if train_dir.exists() and test_dir.exists():
        return [train_dir, test_dir]

    # Untuk dataset yang hanya punya satu folder berisi subfolder class.
    return [dataset_dir]


def has_official_train_test(dataset_dir: str | Path = "dataset") -> bool:
    dataset_dir = Path(dataset_dir)
    return (dataset_dir / "train").exists() and (dataset_dir / "test").exists()


def list_image_paths(class_dir: Path) -> list[Path]:
    if not class_dir.exists():
        return []

    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_first_image(dataset_dir: str | Path = "dataset") -> Path | None:
    dataset_dir = Path(dataset_dir)
    for split_dir in get_split_dirs(dataset_dir):
        for class_name in EMOTION_CLASSES:
            images = list_image_paths(split_dir / class_name)
            if images:
                return images[0]
    return None


def load_dataset(
    split_dir: str | Path,
    debug: bool = False,
    max_images_per_class: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    split_dir = Path(split_dir)

    if not split_dir.exists():
        raise FileNotFoundError(f"Dataset folder tidak ditemukan: {split_dir}")

    X = []
    y = []
    skipped_images = 0

    for class_name in EMOTION_CLASSES:
        class_dir = split_dir / class_name
        image_paths = list_image_paths(class_dir)

        if max_images_per_class is not None:
            image_paths = image_paths[:max_images_per_class]

        print(f"Loading {class_name:8s}: {len(image_paths)} images")

        for image_path in image_paths:
            image = preprocess_image(image_path)
            if image is None:
                skipped_images += 1
                continue

            features = extract_features(image)

            # rows    = jumlah image
            # columns = jumlah features per image
            X.append(features)
            y.append(class_name)

    if skipped_images:
        print(f"Skipped corrupt/unreadable images: {skipped_images}")

    X_array = np.array(X, dtype=np.float32)
    y_array = np.array(y)

    if debug:
        print(f"X shape: {X_array.shape}")
        print(f"y shape: {y_array.shape}")

    return X_array, y_array
