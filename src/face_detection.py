from pathlib import Path

import cv2
import numpy as np


CASCADE_FILENAMES = [
    "haarcascade_frontalface_default.xml",
    "haarcascade_frontalface_alt2.xml",
    "haarcascade_frontalface_alt.xml",
]
CASCADE_DIRS = [
    Path("/usr/share/opencv4/haarcascades"),
    Path("/usr/share/opencv/haarcascades"),
]


def _cascade_candidates(filename: str) -> list[Path]:
    candidates = []

    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        candidates.append(Path(cv2.data.haarcascades) / filename)

    candidates.extend(cascade_dir / filename for cascade_dir in CASCADE_DIRS)
    return candidates


def find_cascade_path() -> str:
    for cascade_path in _cascade_candidates(CASCADE_FILENAMES[0]):
        if cascade_path.exists():
            return str(cascade_path)

    raise RuntimeError(
        "Haar Cascade file tidak ditemukan. Install data OpenCV, contoh: "
        "sudo dnf install opencv"
    )


def load_face_detector() -> list[cv2.CascadeClassifier]:
    detectors = []

    for filename in CASCADE_FILENAMES:
        for cascade_path in _cascade_candidates(filename):
            if not cascade_path.exists():
                continue

            detector = cv2.CascadeClassifier(str(cascade_path))
            if not detector.empty():
                detectors.append(detector)
                break

    if not detectors:
        raise RuntimeError("Haar Cascade face detector gagal dimuat.")

    return detectors


def normalize_lighting(gray_frame: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(gray_frame)


def _overlap_ratio(box_a, box_b) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    x_left = max(ax, bx)
    y_top = max(ay, by)
    x_right = min(ax + aw, bx + bw)
    y_bottom = min(ay + ah, by + bh)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    overlap_area = (x_right - x_left) * (y_bottom - y_top)
    smaller_area = min(aw * ah, bw * bh)
    return overlap_area / smaller_area


def _deduplicate_faces(faces: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    selected = []

    for face in sorted(faces, key=lambda box: box[2] * box[3], reverse=True):
        if all(_overlap_ratio(face, existing) < 0.45 for existing in selected):
            selected.append(face)

    return selected


def detect_faces(gray_frame: np.ndarray, detectors: list[cv2.CascadeClassifier]):
    enhanced_frame = normalize_lighting(gray_frame)
    min_face_size = max(45, min(gray_frame.shape[:2]) // 8)
    faces = []

    for detector in detectors:
        detected = detector.detectMultiScale(
            enhanced_frame,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(min_face_size, min_face_size),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        faces.extend(tuple(map(int, face)) for face in detected)

    return _deduplicate_faces(faces)
