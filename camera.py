from pathlib import Path
from collections import Counter, deque
import sys

import cv2
import joblib
import numpy as np

from src.face_detection import detect_faces, load_face_detector
from src.features import extract_features
from src.preprocessing import preprocess_image


MODEL_PATH = Path("models/emotion_model.pkl")
DEFAULT_CAMERA_INDICES = range(6)
FACE_MARGIN = 0.22
MAX_MISSED_FACE_FRAMES = 10
PREDICTION_HISTORY_SIZE = 9
FRAME_SIZE = (640, 480)
MAX_EMPTY_FRAMES = 30
DETECTION_SCALE = 0.5
DETECT_EVERY_N_FRAMES = 2
PREDICT_EVERY_N_FRAMES = 3


def format_label(label: str) -> str:
    return label.capitalize()


def smooth_box(previous_box, current_box, alpha: float = 0.65):
    if previous_box is None:
        return current_box

    return tuple(
        int(alpha * current_value + (1 - alpha) * previous_value)
        for previous_value, current_value in zip(previous_box, current_box)
    )


def crop_with_margin(gray_frame: np.ndarray, box, margin_ratio: float = FACE_MARGIN):
    x, y, w, h = box
    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(gray_frame.shape[1], x + w + margin_x)
    y2 = min(gray_frame.shape[0], y + h + margin_y)

    return gray_frame[y1:y2, x1:x2]


def largest_face(faces):
    return max(faces, key=lambda box: box[2] * box[3]) if faces else None


def scale_box(box, scale: float):
    x, y, w, h = box
    return (
        int(x / scale),
        int(y / scale),
        int(w / scale),
        int(h / scale),
    )


def stable_prediction(model, features, prediction_history):
    prediction = model.predict([features])[0]
    score = None

    if hasattr(model, "decision_function"):
        scores = model.decision_function([features])
        score = float(np.max(scores))

    prediction_history.append(prediction)
    stable_label = Counter(prediction_history).most_common(1)[0][0]
    return stable_label, score


def configure_camera(camera):
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
    camera.set(cv2.CAP_PROP_FPS, 30)


def parse_camera_source() -> int | str | None:
    if len(sys.argv) <= 1:
        return None

    try:
        return int(sys.argv[1])
    except ValueError:
        return sys.argv[1]


def open_camera():
    requested_source = parse_camera_source()
    if requested_source is not None:
        camera_sources = [requested_source]
    else:
        device_paths = sorted(str(path) for path in Path("/dev").glob("video*"))
        camera_sources = [*device_paths, *DEFAULT_CAMERA_INDICES]

    for camera_source in camera_sources:
        camera = cv2.VideoCapture(camera_source)
        if camera.isOpened():
            configure_camera(camera)
            print(f"Using camera: {camera_source}")
            return camera, camera_source

        camera.release()

        camera = cv2.VideoCapture(camera_source, cv2.CAP_V4L2)
        if camera.isOpened():
            configure_camera(camera)
            print(f"Using camera: {camera_source}")
            return camera, camera_source

        camera.release()

    print("Webcam tidak tersedia.")
    print("Cek device kamera dengan:")
    print("v4l2-ctl --list-devices")
    print("Lalu jalankan, contoh:")
    print("python camera.py 1")
    print("atau:")
    print("python camera.py /dev/video1")
    return None, None


def reopen_camera(camera_source):
    camera = cv2.VideoCapture(camera_source)
    if camera.isOpened():
        configure_camera(camera)
        return camera

    camera.release()
    return None


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

    camera, camera_source = open_camera()
    if camera is None:
        return

    print("Webcam started. Tekan q untuk keluar.")
    last_face = None
    missed_face_frames = 0
    empty_frame_count = 0
    frame_count = 0
    current_label = None
    current_score = None
    prediction_history = deque(maxlen=PREDICTION_HISTORY_SIZE)

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                empty_frame_count += 1
                if empty_frame_count < MAX_EMPTY_FRAMES:
                    cv2.waitKey(30)
                    continue

                print("Frame webcam tidak dapat dibaca. Mencoba buka ulang kamera...")
                camera.release()
                camera = reopen_camera(camera_source)
                empty_frame_count = 0
                if camera is None:
                    print("Kamera gagal dibuka ulang.")
                    break
                continue

            empty_frame_count = 0
            frame_count += 1

            frame = cv2.resize(frame, FRAME_SIZE)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            should_detect = frame_count % DETECT_EVERY_N_FRAMES == 0 or last_face is None
            current_face = None
            if should_detect:
                detection_frame = cv2.resize(gray_frame, None, fx=DETECTION_SCALE, fy=DETECTION_SCALE)
                faces = detect_faces(detection_frame, face_detector)
                detected_face = largest_face(faces)
                current_face = scale_box(detected_face, DETECTION_SCALE) if detected_face else None

            if current_face is not None:
                last_face = smooth_box(last_face, current_face)
                missed_face_frames = 0
            elif last_face is not None and (
                not should_detect or missed_face_frames < MAX_MISSED_FACE_FRAMES
            ):
                missed_face_frames += 1
            else:
                last_face = None
                prediction_history.clear()
                missed_face_frames = 0
                current_label = None
                current_score = None

            if last_face is not None:
                x, y, w, h = last_face
                should_predict = (
                    frame_count % PREDICT_EVERY_N_FRAMES == 0
                    or current_label is None
                    or current_face is not None
                )

                if should_predict:
                    face_crop = crop_with_margin(gray_frame, last_face)
                    face_image = preprocess_image(face_crop, equalize_hist=True)
                    if face_image is not None:
                        features = extract_features(face_image)
                        current_label, current_score = stable_prediction(
                            model, features, prediction_history
                        )

                label = "Detecting..."
                if current_label is not None:
                    label = f"Facial Expression: {format_label(current_label)}"
                    if current_score is not None:
                        label = f"{label} | Score: {current_score:.2f}"

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
            else:
                cv2.putText(
                    frame,
                    "Face not detected",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
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
