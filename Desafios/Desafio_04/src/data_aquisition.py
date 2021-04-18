import os

import cv2 as cv
import numpy as np
from utils import make_sure_path_exists
from variables import (ALLOWED_METHODS, CLASSIFIER_MIN_SZ, CLASSIFIER_NEIGHBORS, CLASSIFIER_SCALE, FACE_BOX_COLOR,
                       HEURISTIC_FILE_PREFIX, KNOWN_USER_FILE, LBPH_MACHINE_LEARNING_FILE, LIB_FACE_RECOGNITION_FILE_PREFIX,
                       MIN_SAMPLES_PER_USER)


class DataAquisition:
    """Based on: https://github.com/Harmouch101/Face-Recogntion-Detection"""
    def __init__(
        self,
        face_detector_name: str = 'haarcascade_frontalface_alt.xml',
        right_eye_detector_name: str = 'haarcascade_lefteye_2splits.xml',
        left_eye_detector_name: str = 'haarcascade_righteye_2splits.xml',
        dataset_dir: str = 'dataset',
        resources_dir: str = 'resources',
        cnt_samples: int = MIN_SAMPLES_PER_USER
    ) -> None:
        self.face_detector = cv.CascadeClassifier(cv.data.haarcascades + face_detector_name)
        self.right_eye_detector_name = cv.CascadeClassifier(cv.data.haarcascades + right_eye_detector_name)
        self.left_eye_detector_name = cv.CascadeClassifier(cv.data.haarcascades + left_eye_detector_name)

        self.dataset_dir = make_sure_path_exists(dataset_dir)
        self.resources_dir = make_sure_path_exists(resources_dir)
        self.users_file = make_sure_path_exists(os.path.join(self.resources_dir, KNOWN_USER_FILE), False)

        self.cnt_samples = cnt_samples

    def create_dataset(self, webcam):
        while self._is_creating_dataset():
            print('Capturing person images...', )
            dataset_dir = make_sure_path_exists(os.path.join(self.dataset_dir, str(self.user_id), ' '))
            cnt = 0

            while cnt < self.cnt_samples:
                _, c_frame = webcam.read()
                c_gray_frame = cv.cvtColor(c_frame, cv.COLOR_BGR2GRAY)

                face = self.face_detector.detectMultiScale(
                    c_gray_frame, CLASSIFIER_SCALE, CLASSIFIER_NEIGHBORS, minSize=CLASSIFIER_MIN_SZ
                )

                if len(face) > 1:
                    print('There are more than one face detected')
                    continue

                try:
                    x, y, w, h = face[0]
                    cv.rectangle(c_frame, (x, y), (x + w, y + h), FACE_BOX_COLOR, 2)

                    outer_face = c_gray_frame[
                        y - CLASSIFIER_MIN_SZ[1]: y + h + CLASSIFIER_MIN_SZ[1],
                        x - CLASSIFIER_MIN_SZ[0]: x + w + CLASSIFIER_MIN_SZ[0],
                    ]
                    face = c_gray_frame[y:y + h, x: x +w]

                    rx, ry, rw, rh = self._eyes_location(face[y: y + int(h // 2), x: x + int(w // 2)], 'right')
                    lx, ly, lw, lh = self._eyes_location(face[y: y + int(h // 2), x + int(w // 2): x + w], 'left')

                    cv.rectangle(c_frame, (rx + x, ry + y), (rx + rw + x, ry + rh + y), (255, 255, 255), 2)
                    cv.rectangle(c_frame, (lx + x, ly + y), (lx + lw + x, ly + lh + y), (255, 0, 0), 2)

                    angle = self._angle_between_eyes((rx, ry, rw, rh), (lx, ly, lw, lh))
                    outer_face = self._rotate_face(outer_face, angle)

                    cv.imwrite(os.path.join(dataset_dir, f'img{cnt}.jpg'), outer_face)

                    cnt += 1
                except Exception as e:
                    print(f'\nSomething went wrong: {e}')

                cv.imshow('webcam', c_frame)
                cv.waitKey(1)

        return self._parse_user_ids()

    def train(self, method: str = 'lbph_machinelearning'):
        img_faces, img_labels = self._get_labeled_images()

        if method not in ALLOWED_METHODS:
            return

        exclusive_method = getattr(self, f'_train_{method}', None)
        if exclusive_method:
            return exclusive_method(img_faces, img_labels)

        return self._train_generic(img_faces, img_labels, method)

    def _is_creating_dataset(self):
        new_name = input('Person name to register or 0 to continue:\n')
        self.user_id = -1

        if new_name == '0':
            return False

        self.user_id, max_user_id = self._get_know_user_id(new_name)

        if self.user_id != -1:
            return True

        with open(self.users_file, 'a+') as user_file:
            self.user_id = max_user_id + 1

            user_file.write(f'{self.user_id},{new_name}\n')

        return True

    def _get_know_user_id(self, name: str) -> tuple:
        max_user_id = -1

        with open(self.users_file, 'r+') as user_file:
            user_file.seek(0)

            for line in user_file.readlines():
                user_id, user_name = line.strip().split(',')

                if user_name == name:
                    return user_id, max_user_id

                max_user_id = max(max_user_id, int(user_id))

        return -1, max_user_id

    def _parse_user_ids(self) -> dict:
        users = {-1: 'unknown'}

        with open(self.users_file, 'r') as user_file:
            for line in user_file.readlines():
                user_id, user_name = line.strip().split(',')

                users[int(user_id)] = user_name

        return users

    def _eyes_location(self, face, which_eye: str = 'right'):
        eyes = getattr(self, f'{which_eye}_eye_detector_name').detectMultiScale(
            face, CLASSIFIER_SCALE * .95, CLASSIFIER_NEIGHBORS
        )

        if len(eyes) > 1:
            raise Exception(f'There are more eyes than expected: {which_eye}')

        if not np.any(eyes):
            print(f'There are not eyes: {which_eye}')
            return 0, 0, 0, 0

        return eyes[0]

    def _angle_between_eyes(self, right_eye, left_eye):
        right_center = (right_eye[0] + right_eye[3] / 2, right_eye[1] + right_eye[2] / 2)
        left_center = (left_eye[0] + left_eye[3] / 2, left_eye[1] + left_eye[2] / 2)

        try:
            rad = np.arctan(abs(right_center[0] - left_center[0]) / abs(right_center[1] - left_center[1]))

            return rad * 180 / np.pi
        except:
            return 0

    def _rotate_face(self, img_face, angle):
        if not angle:
            return img_face

        face_center = tuple(np.array(img_face.shape) / 2)
        rot_mat = cv.getRotationMatrix2D(face_center, angle, 1.)

        return cv.warpAffine(img_face, rot_mat, img_face.shape, flags=cv.INTER_LINEAR)

    def _get_labeled_images(self):
        labeled_images, labels = [], []

        for c_dir, _, file_names in os.walk(self.dataset_dir):
            label = c_dir.replace(self.dataset_dir + '/', '')

            for fn in file_names:
                c_img = cv.imread(os.path.join(c_dir, fn), cv.IMREAD_GRAYSCALE)
                labeled_images.append(np.array(c_img, np.uint8))
                labels.append(int(label))

        return labeled_images, np.array(labels)

    def _train_lbph_machinelearning(self, images, labels):
        training_file = os.path.join(self.resources_dir, LBPH_MACHINE_LEARNING_FILE)

        from variables import LBP_GRID_X, LBP_GRID_Y, LBP_NEIGHBORS, LBP_RADIUS

        recognizer = cv.face.LBPHFaceRecognizer_create(
            LBP_RADIUS, LBP_NEIGHBORS, LBP_GRID_X, LBP_GRID_Y
        )

        recognizer.update(images, labels)
        recognizer.write(training_file)

    def _train_generic(self, images, raw_labels, method):
        file_prefix = HEURISTIC_FILE_PREFIX if method == 'heuristic' else LIB_FACE_RECOGNITION_FILE_PREFIX

        descriptors_file = os.path.join(self.resources_dir, f'{file_prefix}_desc.npy')
        labels_file = os.path.join(self.resources_dir, f'{file_prefix}_labels.npy')

        descriptors, labels = [], []
        for ind, image in enumerate(images):
            descriptor = getattr(self, f'_get_{method}_descriptor')(image)

            if not np.any(descriptor):
                continue

            descriptors.append(descriptor if not isinstance(descriptor, list) else descriptor[0])
            labels.append(raw_labels[ind])

        descriptors = np.asarray(descriptors)
        labels = np.asarray(labels)

        np.save(descriptors_file, descriptors)
        np.save(labels_file, labels)

    def _get_heuristic_descriptor(self, image):
        from local_binary_pattern import LocalBinaryPatterns

        descriptor, _ = LocalBinaryPatterns().get_descriptor(image)

        return descriptor

    def _get_lib_face_recognition_descriptor(self, image):
        from face_recognition import face_encodings

        img = cv.cvtColor(image, cv.COLOR_GRAY2RGB)

        return face_encodings(img)
