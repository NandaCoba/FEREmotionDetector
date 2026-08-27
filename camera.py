from pathlib import Path

import cv2
import joblib
import numpy as np

from src.face_detection import detect_faces, load_face_detector
from src.features import extract_features
from src.preprocessing import preprocess_image


MODEL_PATH = Path("models/emotion_model.pkl")


def format_label(label: str) -> str:
    return label.capitalize()


def main():
    if not MODEL_PATH.exists():
        print("Model not found.")
        print("Run:")
        print("python train.py")
        return

    saved = joblib.load(MODEL_PATH)
    model = saved["model"] if isinstance(saved, dict) else saved

    try:
        face_detector = load_face_detector()
    except RuntimeError as error:
        print(error)
        return

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Webcam tidak tersedia.")
        return

    print("Webcam started. Tekan q untuk keluar.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Frame webcam tidak dapat dibaca.")
                break

            frame = cv2.resize(frame, (640, 480))
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(gray_frame, face_detector)

            for x, y, w, h in faces:
                face_crop = gray_frame[y : y + h, x : x + w]
                face_image = preprocess_image(face_crop)
                if face_image is None:
                    continue

                features = extract_features(face_image)
                prediction = model.predict([features])[0]

                label = f"Facial Expression: {format_label(prediction)}"

                if hasattr(model, "decision_function"):
                    scores = model.decision_function([features])
                    score = np.max(scores)
                    label = f"{label} | Score: {score:.2f}"

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x, max(25, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("FEREmotionDetector", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
