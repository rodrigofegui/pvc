import cv2
import numpy as np
from utils import get_background_image, stackImages
from virtual_background import background_subtraction, contour_detection, face_detection


def main(processing_method):
    webcam = cv2.imread('../resources/câmera_1.png', cv2.IMREAD_COLOR)

    background_img = get_background_image(webcam)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        _, cur_frame = webcam.read()

        mask = processing_method(cur_frame)

        masked_frame = cv2.bitwise_and(cur_frame, cur_frame, mask=mask)
        indexes = np.where(masked_frame == 0)
        masked_frame[indexes] = background_img[indexes]

        stacked_img = stackImages(
            .6, (
                [cur_frame, background_img],
                [mask, masked_frame]
            )
        )

        cv2.imshow("Imagens", stacked_img)

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(background_subtraction)
