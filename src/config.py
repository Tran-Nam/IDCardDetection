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

NUM_CLASSES = 4
LABEL_MAP = {
    0: 'cmnd_front',
    1: 'cmnd_back',
    2: 'cccd_front',
    3: 'cccd_back'
}

### TRAINING PARAM
BATCH_SIZE = 4
PRETRAINED = True 
PRETRAINED_PATH = '../checkpoint_10/58000.ckpt'

LEARNING_RATE = 1e-4
DECAY_RATE = 0.95
DECAY_STEP = 20000

INTERVAL_SAVE = 1000
MODEL_DIR = '../checkpoint_10'


