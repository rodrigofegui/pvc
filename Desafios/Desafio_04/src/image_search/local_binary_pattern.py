import cv2 as cv
import numpy as np
from skimage import feature


class LocalBinaryPatterns:
    """Based on: https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/"""
    def __init__(self) -> None:
        self._num_points = 16
        self._radius = 2

    def cvt_image(self, src: np.ndarray) -> np.ndarray:
        if len(src.shape) == 3:
            src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

        return feature.local_binary_pattern(src, self._num_points, self._radius, method='uniform')

    def get_descriptor(self, src: np.ndarray, eps: float=1e-7):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
        lbp = self.cvt_image(src)

        descriptor, _ = np.histogram(
            lbp.ravel(),
            bins=np.arange(0, self._num_points + 3),
            range=(0, self._num_points + 2)
        )
        # normalize the histogram
        descriptor = descriptor.astype('float')
        descriptor /= (descriptor.sum() + eps)
        # return the histogram of Local Binary Patterns
        return descriptor, lbp
