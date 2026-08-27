import cv2
import numpy as np


def load_face_detector() -> cv2.CascadeClassifier:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        raise RuntimeError("Haar Cascade face detector gagal dimuat.")

    return detector


def detect_faces(gray_frame: np.ndarray, detector: cv2.CascadeClassifier):
    return detector.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
