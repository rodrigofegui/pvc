# Base dirs to Challenge #1
WORKDIRS_C1 = [
    '../data/Middlebury/Jadeplant-perfect',
    '../data/Middlebury/Playtable-perfect',
]

# Base dirs to Challenge #2
WORKDIRS_C2 = [
    '../data/FurukawaPonce/Morpheus',
    '../data/FurukawaPonce/warrior',
]

# Dir to save the result images
RESULT_DIR = '../results'

# Lowest dimensions in image resize
IMG_LOWEST_DIMENSION = 350

# Minimum filter size to calculate disparity map
MIN_DISP_FILTER_SZ = 3
# Maximum filter size to calculate disparity map
MAX_DISP_FILTER_SZ = 17
# Default search window to calculate disparity map
DISP_BLOCK_SEARCH = 64

# Threshold used at disparity map error comparison
ERROR_THRESHOLD = 2

# Resize control
SHOULD_RESIZE = True

# Basic post processing median blur size
MEDIAN_FLT_SZ = 5
