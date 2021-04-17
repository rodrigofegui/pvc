from skimage import feature
import numpy as np

class LocalBinaryPatterns:
    """Based on: https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/"""
    def __init__(self, num_points: int = 1, radius: int=1) -> None:
        self._num_points = num_points
        self._radius = radius

    def cvt_image(self, src: np.ndarray) -> np.ndarray:
        return feature.local_binary_pattern(src, self._num_points, self._radius, method='uniform')

    def get_descriptor(self, src: np.ndarray, eps: float=1e-7):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
        lbp = self.cvt_image(src)

        hist, _ = np.histogram(
            lbp.ravel(),
            bins=np.arange(0, self.numPoints + 3),
            range=(0, self.numPoints + 2)
        )
        # normalize the histogram
        hist = hist.astype('float')
        hist /= (hist.sum() + eps)
        # return the histogram of Local Binary Patterns
        return hist
