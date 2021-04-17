import cv2 as cv
import numpy as np

from .face_detection import get_lib_face_recognition_features, lib_face_recognition


def main(face_recognition_method):
    if face_recognition_method == lib_face_recognition:
        known_encoders, known_names = get_lib_face_recognition_features('images/training/')
    else:
        known_encoders, known_names = [], []

    webcam = cv.VideoCapture(0)

    while True:
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        _, c_frame = webcam.read()

        c_frame = face_recognition_method(c_frame, known_encoders, known_names)

        cv.imshow('webcam', c_frame)

    webcam.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main(lib_face_recognition)
