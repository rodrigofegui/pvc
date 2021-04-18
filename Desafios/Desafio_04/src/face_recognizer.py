import os

import cv2 as cv
import numpy as np
from utils import make_sure_path_exists
from variables import (ALLOWED_METHODS, CLASSIFIER_NEIGHBORS, CLASSIFIER_SCALE, FACE_BOX_COLOR, FACE_TXT_COLOR,
                       HEURISTIC_CERTAIN, HEURISTIC_FILE_PREFIX, LIB_FACE_RECOGNITION_FILE_PREFIX)


class FaceRecognizer:
    """Based on: https://github.com/Harmouch101/Face-Recogntion-Detection"""
    def __init__(
        self,
        resources_dir: str = 'resources',
        recognizer_name: str = 'lbph_machinelearning',
        face_detector_name: str = 'haarcascade_frontalface_alt.xml',
    ) -> None:
        self.resources_dir = make_sure_path_exists(resources_dir)

        self.face_detector = cv.CascadeClassifier(cv.data.haarcascades + face_detector_name)

        self._set_recognizer(recognizer_name)

    def predict(self, image: np.ndarray, labels_translation: dict = {}) -> tuple:
        face_locations, labels, factors = getattr(self, f'_prediction_{self._recognizer_name}')(image)

        if labels_translation:
            for ind, (x1, y1, x2, y2) in enumerate(face_locations):
                cv.rectangle(image, (x1, y1), (x2, y2), FACE_BOX_COLOR, 2)

                cv.putText(
                    image,
                    f'{labels_translation[labels[ind]]}: {factors[ind]}',
                    (x1, y1), cv.FONT_HERSHEY_COMPLEX, .7, FACE_TXT_COLOR, 1, cv.LINE_AA
                )

        return image, face_locations, labels, factors

    def _set_recognizer(self, recognizer_name: str) -> None:
        if recognizer_name not in ALLOWED_METHODS:
            raise Exception('Reconhecimento de face desconhecido')

        self._recognizer_name = recognizer_name

        if recognizer_name == 'lbph_machinelearning':
            from variables import LBP_GRID_X, LBP_GRID_Y, LBP_NEIGHBORS, LBP_RADIUS, LBPH_MACHINE_LEARNING_FILE

            self.recognizer = cv.face.LBPHFaceRecognizer_create(
                LBP_RADIUS, LBP_NEIGHBORS, LBP_GRID_X, LBP_GRID_Y
            )
            self.recognizer.read(os.path.join(self.resources_dir, LBPH_MACHINE_LEARNING_FILE))

            return

        file_prefix = HEURISTIC_FILE_PREFIX if recognizer_name == 'heuristic' else LIB_FACE_RECOGNITION_FILE_PREFIX

        descriptors_file = os.path.join(self.resources_dir, f'{file_prefix}_desc.npy')
        labels_file = os.path.join(self.resources_dir, f'{file_prefix}_labels.npy')

        self.known_descriptors = np.load(descriptors_file, allow_pickle=True)
        self.known_labels = np.load(labels_file, allow_pickle=True)

    def _prediction_lib_face_recognition(self, image: np.ndarray) -> tuple:
        """Face recognition using the homonymous module, which implements the HOG
        (Histogram of Oriented Gradient). The detected faces are marked within a square
        and labeled as `face_name` or `'unknown'`

        Args:
        - `base_image:np.ndarray`: Base image to face recognition
        - `known_encoders:np.array`: Known face encoders to recognition match search
        - `known_names:np.array`: Known face names to recognition match search

        Returns:
        - Marked image
        """
        import face_recognition
        face_locations = face_recognition.face_locations(image)
        c_encoded_faces = face_recognition.face_encodings(image, face_locations)
        labels, factors = [], []

        for ind, encoded_face in enumerate(c_encoded_faces):
            comparisons = face_recognition.compare_faces(self.known_descriptors, encoded_face)
            face_dist = face_recognition.face_distance(self.known_descriptors, encoded_face)
            min_dist_at = np.argmin(face_dist)

            y1, x2, y2, x1 = face_locations[ind]
            face_locations[ind] = (x1, y1, x2, y2)

            labels.append(self.known_labels[min_dist_at] if comparisons[min_dist_at] else -1)

            factors.append(face_dist[min_dist_at])

        return face_locations, labels, factors

    def _prediction_heuristic(self, image):
        from local_binary_pattern import LocalBinaryPatterns

        face_locations, labels, factors = [], [], []

        if len(image.shape) == 3:
            image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        for x, y, w, h in self.face_detector.detectMultiScale(image_gray, CLASSIFIER_SCALE, CLASSIFIER_NEIGHBORS):
            face_locations.append((x, y, x + w, y + h))

            face_box = image_gray[y:y + h, x: x+w]

            descriptor, _ = LocalBinaryPatterns().get_descriptor(face_box)

            face_dist = np.linalg.norm(self.known_descriptors - descriptor, axis=1)
            min_dist_at = np.argmin(face_dist)

            diff = face_dist[min_dist_at] / np.linalg.norm(descriptor)
            print('diff', diff)

            labels.append(self.known_labels[min_dist_at] if diff <= HEURISTIC_CERTAIN else -1)

            factors.append(face_dist[min_dist_at])

        return face_locations, labels, factors

    def _prediction_lbph_machinelearning(self, image):
        face_locations, labels, factors = [], [], []

        if len(image.shape) == 3:
            image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        for x, y, w, h in self.face_detector.detectMultiScale(image_gray, CLASSIFIER_SCALE, CLASSIFIER_NEIGHBORS):
            x1, y1, x2, y2 = x, y, x + w, y + h

            face_locations.append((x1, y1, x2, y2))

            label, factor = self.recognizer.predict(image_gray[y1: y2, x1: x2])

            labels.append(label if factor >= 100 else -1)
            factors.append(factor)

        return face_locations, labels, factors
