import numpy as np 
from pafprocess import paf_to_idcard

def decode(heatmap, paf, offset, ratio=4):
    """
    find location of 4 corner from output of the net
    :heatmap - joints heatmap
    :paf - part affinity field heatmap
    :offset - offset of point
    return 
    """
