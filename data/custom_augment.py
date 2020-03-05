import numpy as np 

def RandomCrop(im, list_pts):
    State = np.random.RandomState()
    im_h, im_w = im.shape[:2]
    list_pts = np.array(list_pts).reshape(-1, 4, 2)
    xmin, ymin = np.min(list_pts, axis=(0, 1))
    xmax, ymax = np.max(list_pts, axis=(0, 1))
    begin_x, begin_y = State.randint(0, xmin), State.randint(0, ymin)
    end_x, end_y = State.randint(xmax, im_w), State.randint(ymax, im_h)
    im_crop = im[begin_y: end_y, begin_x: end_x, :]
    pts_crop = list_pts - np.array([xmin, ymin])
    return im_crop, pts_crop
    
