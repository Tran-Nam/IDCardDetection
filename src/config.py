import numpy as np
### MODEL PARAM
NUM_JOINTS = 4
LIMBS = [
    [0, 1],
    [1, 2],
    [2, 3], 
    [3, 0]
]
NUM_LIMBS = 4
PAF_XY_COORDS_PER_LIMB = np.arange(8).reshape(4, 2)

### TRAINING PARAM
BATCH_SIZE = 4
PRETRAINED = False 
PRETRAINED_PATH = ''
TRAIN_DATA_PATH = ''
VAL_DATA_PATH = ''

LEARNING_RATE = 1e-4
INTERVAL_SAVE = 1000
MODEL_DIR = '../checkpoint'

