import numpy as np 

def normalize(v):
    """
    normalize a vector to have norm = 1
    """
    return v / np.linalg.norm(v)

def find_direction_vector(line):
    """
    find normalized direction vector  of a line
    :param line has format: [pt1, pt2]
    """
    pt1, pt2 = line
    pt1 = np.array(pt1).reshape(2,)
    pt2 = np.array(pt2).reshape(2,)
    direct = pt2 - pt1
    direct_norm = normalize(direct)
    return direct_norm

def find_perpendicular_vector(vt):
    """
    find vector perpenticular with given vector
    """
    x, y = vt
    return np.array([y, -x])

def cal_length_line(line):
    pt1, pt2 = line
    return ((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)**0.5

def cal_length_vector(vt):
    return np.linalg.norm(vt)

def convert_coor_im2decart(pts, w, h):
    """
    convert corrdinate of pts from image to decart
    :param pts: list of points
    :param w: image width
    :param h: image height
    """
    return [[x, h-1-y] for x, y in pts]