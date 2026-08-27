from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.dataset_loader import EMOTION_CLASSES, has_official_train_test, load_dataset
from src.features import HOG_PARAMS
from src.preprocessing import IMAGE_SIZE


DATASET_DIR = Path("dataset")
MODEL_PATH = Path("models/emotion_model.pkl")
RANDOM_STATE = 42


def main():
    if not DATASET_DIR.exists():
        print("Dataset folder tidak ditemukan.")
        print("Letakkan FER-2013 di folder dataset/.")
        return

    if has_official_train_test(DATASET_DIR):
        print("Using official FER-2013 train/test folders.")
        print()

        X_train, y_train = load_dataset(DATASET_DIR / "train", debug=True)
        print()
        X_test, y_test = load_dataset(DATASET_DIR / "test", debug=True)
    else:
        print("Official train/test folders tidak ditemukan.")
        print("Using train_test_split(test_size=0.2, stratify=y).")
        print()

        X, y = load_dataset(DATASET_DIR, debug=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y,
        )

    print()
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape : {X_test.shape}")
    print(f"y_test shape : {y_test.shape}")
    print()

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LinearSVC(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    max_iter=5000,
                    dual="auto",
                ),
            ),
        ]
    )

    print("Training started...")
    print()
    print(f"Training samples : {X_train.shape[0]}")
    print(f"Features         : {X_train.shape[1]}")
    print(f"Classes          : {len(EMOTION_CLASSES)}")
    print()

    model.fit(X_train, y_train)

    print("Training completed.")
    print()

    y_pred = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred, labels=EMOTION_CLASSES, zero_division=0))
    print("Confusion Matrix:")
    print(EMOTION_CLASSES)
    print(confusion_matrix(y_test, y_pred, labels=EMOTION_CLASSES))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "image_size": IMAGE_SIZE,
            "hog_params": HOG_PARAMS,
            "classes": EMOTION_CLASSES,
        },
        MODEL_PATH,
    )

    print()
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
