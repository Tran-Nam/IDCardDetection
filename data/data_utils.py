import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
from scipy.spatial import distance as dist

def imshow(im):
    if len(im.shape)==3:
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(im, cmap='gray')
    plt.show()

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def order_points(pts):
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    return np.array([tl, tr, br, bl], dtype="int")

def rotate_box(bb, cx, cy, h, w, theta):
    new_bb = list(bb)
    for i,coord in enumerate(bb):
        M = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cx
        M[1, 2] += (nH / 2) - cy
        v = [coord[0],coord[1],1]
        calculated = np.dot(M,v).astype(int)
        new_bb[i] = [calculated[0],calculated[1]]
    return new_bb

def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH))

def get_pts(im):
    assert im.shape[2] == 4
    im_alpha = im[:, :, 3].astype(np.uint8)
    kernel = np.ones((5,5), np.uint8)
    im_alpha = cv2.erode(im_alpha, kernel, iterations=2) # erosion box
    cnts, _ = cv2.findContours(im_alpha, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)[0] # get max area cnt
    epsilon = 0.1*cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    if approx.shape[0] != 4:
        return None
    approx = order_points(approx.reshape(4, 2))
    return approx

def calcLineLength(point1, point2):
    x1, y1 = point1 
    x2, y2 = point2 
    length = ((x1-x2)**2+(y1-y2)**2)**0.5
    return length

def calcRadius(size, iou_thresh=0.7): # implement base on origin paper
    h, w = size
    l = (h**2+w**2)**0.5

    # det inside
    a1 = 4
    b1 = -4*l
    c1 = -(1-iou_thresh)*l**2
    delta1 = b1**2-4*a1*c1
    r1 = (-b1 + delta1**0.5) / (2*a1)

    # det outside
    a2 = 4
    b2 = 4*l
    c2 = (1-1/iou_thresh)*l**2
    delta2 = b2**2 - 4*a2*c2
    r2 = (-b2 + delta2**0.5) / (2*a2)

    # det cross
    r3 = (((1-iou_thresh)*l**2)/4)**0.5

    return int(min(r1, r2, r3))

def gaussian2D(size, sigma=1): # normalize range [0, 1]
    h, w = size
    gaussian = np.zeros((h, w))
    center = [h//2, w//2]
    for i in range(h):
        for j in range(w):
            gaussian[i, j] = 1 / (2*np.pi*sigma**2) * np.exp(-((i-center[0])**2+(j-center[1])**2)/(2*sigma**2))
    # print(gaussian.shape)
    gaussian = gaussian / np.max(gaussian)
    return gaussian

def draw_gaussian(heatmap, center, radius):
    diameter = 2*radius + 1
    sigma = diameter / 6 # in paper
    gaussian = gaussian2D((diameter, diameter), sigma=sigma)

    center_x, center_y = center
    h, w = heatmap.shape[0:2]
    
    left = min(center_x, radius)
    right = min(w-center_x, radius+1)
    top = min(center_y, radius)
    bottom = min(h-center_y, radius+1)

    # print(heatmap[:5, :5])
    mask_gaussian = gaussian[radius-top: radius+bottom, radius-left: radius+right]
    mask_heatmap = heatmap[center_y-top: center_y+bottom, center_x-left: center_x+right]
    np.maximum(mask_heatmap, mask_gaussian, out=mask_heatmap)

def getSizePolygon(pts):
    pts = order_points(pts)
    tl, tr, br, bl = pts 
    side_left = cal_length_line([tl, bl])
    side_top = cal_length_line([tl, tr])
    side_right = cal_length_line([tr, br])
    side_bottom = cal_length_line([br, bl])
    size = (max(side_left, side_right), max(side_top, side_bottom))
    return size

def findMax2d(x):
    """
    find index of max value in 2d array
    """
    m, n = x.shape 
    x_ = x.ravel()
    idx = np.argmax(x_)
    i = idx // n 
    j = idx % n 
    return i, j

def sigmoid(x):
    return 1/(1+np.exp(-x))

def resize_box_and_im(im, pts, size=(512, 512)):
    im_h, im_w = im.shape[:2]
    new_h, new_w = size
    ratio_x = new_w / im_w 
    ratio_h = new_h / im_h 
    new_im = cv2.resize(im, size)
    ratio = np.expand_dims(np.array([ratio_x, ratio_h]), axis=0)
    new_pts = (np.array(pts).reshape(-1, 4, 2) * ratio).astype('int')
    return new_im, new_pts

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

def resize(im, list_pts, side=512): # resize image keep aspect ratio, padding 0
    """
    :param im: origin image
    :param pts:  tl, tr, br, bl: 4*2
    :param side: new image size
    return im and boxes after resize
    """
#     print(list_pts.shape)
    im_h, im_w = im.shape[:2]
    ratio = side / max(im_h, im_w)
    new_size = (int(im_h*ratio), int(im_w*ratio))
    new_im = cv2.resize(im, None, fx=ratio, fy=ratio)
    list_pts = np.array(list_pts).reshape(-1, 4, 2).astype('float32')
    for i in range(len(list_pts)):
        list_pts[i] = list_pts[i].astype('float32')
        list_pts[i][:, 0] *= ratio
        list_pts[i][:, 1] *= ratio

    big_image = np.zeros([side, side, 3], dtype='float32')
    padding_y = (side - new_im.shape[0])//2
    padding_x = (side - new_im.shape[1])//2
    big_image[padding_y: padding_y+new_im.shape[0],
        padding_x: padding_x+new_im.shape[1]] = new_im
    big_image = big_image.astype('uint8')

    for i in range(len(list_pts)):
        list_pts[i][:, 0] += padding_x
        list_pts[i][:, 1] += padding_y
        list_pts[i] = list_pts[i].astype('int')

    return big_image, list_pts