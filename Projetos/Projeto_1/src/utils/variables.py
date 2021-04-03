# Original image dir
WORKDIRS = [
    '../data/Middlebury/Jadeplant-perfect',
    '../data/Middlebury/Playtable-perfect',
]

# Dir to save the result images
RESULT_DIR = '../results'

# Lowest dimensions in image resize
IMG_LOWEST_DIMENSION = 350

# Minimum filter size to calculate disparity map
MIN_DISP_FILTER_SZ = 5
# Default search window to calculate disparity map
DISP_BLOCK_SEARCH = 64

# Threshold used at disparity map error comparison
ERROR_THRESHOLD = 2

# Resize control
SHOULD_RESIZE = False

# Basic post processing median blur size
MEDIAN_FLT_SZ = 5
