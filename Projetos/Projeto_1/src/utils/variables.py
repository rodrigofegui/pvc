# Base dirs to Challenge #1
WORKDIR_C1 = '../data/Middlebury/'

# Base dirs to Challenge #2
WORKDIR_C2 = '../data/FurukawaPonce/'

# Dir to save the result images
RESULT_DIR = '../results'

# Lowest dimensions in image resize
IMG_LOWEST_DIMENSION = 480

# Minimum filter size to calculate disparity map
MIN_DISP_FILTER_SZ = 3
# Maximum filter size to calculate disparity map
MAX_DISP_FILTER_SZ = 17

# Minimum look up size to calculate disparity map
MIN_DISP_LOOKUP = 400
# Maximum look up size to calculate disparity map
MAX_DISP_LOOKUP = 800

# Threshold used at disparity map error comparison
ERROR_THRESHOLD = 2

# Resize control
SHOULD_RESIZE = False

# Basic post processing median blur size
MEDIAN_FLT_SZ = 5

# Best parse points match threshold
BEST_MATCH_PERC = .9
# Maximun matches per points
MATCHES_PER_PT = 2
FLANN_INDEX_KDTREE = 1
