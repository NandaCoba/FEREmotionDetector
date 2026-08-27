from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.dataset_loader import EMOTION_CLASSES, has_official_train_test, load_dataset


DATASET_DIR = Path("dataset")
MODEL_PATH = Path("models/emotion_model.pkl")


def main():
    if not MODEL_PATH.exists():
        print("Model not found.")
        print("Run:")
        print("python train.py")
        return

    if not DATASET_DIR.exists():
        print("Dataset folder tidak ditemukan.")
        print("Letakkan FER-2013 di folder dataset/.")
        return

    saved = joblib.load(MODEL_PATH)
    model = saved["model"] if isinstance(saved, dict) else saved

    if has_official_train_test(DATASET_DIR):
        print("Evaluating on official test folder.")
        X_test, y_test = load_dataset(DATASET_DIR / "test", debug=True)
    else:
        print("Folder dataset/test tidak ditemukan.")
        print("Untuk evaluasi terpisah, buat folder dataset/test atau jalankan python train.py.")
        return

    y_pred = model.predict(X_test)

    print()
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred, labels=EMOTION_CLASSES, zero_division=0))
    print("Confusion Matrix:")
    print(EMOTION_CLASSES)
    print(confusion_matrix(y_test, y_pred, labels=EMOTION_CLASSES))


if __name__ == "__main__":
    main()
