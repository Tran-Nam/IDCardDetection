import numpy as np 

def RandomCrop(im, list_pts, p=0.9):
    State = np.random.RandomState()
    if State.rand() > p:
        return im, list_pts
    im_h, im_w = im.shape[:2]
    list_pts = np.array(list_pts).reshape(-1, 4, 2)
    xmin, ymin = np.min(list_pts, axis=(0, 1)) 
    xmax, ymax = np.max(list_pts, axis=(0, 1))
    begin_x, begin_y = State.randint(0, xmin+1), State.randint(0, ymin+1)
    end_x, end_y = State.randint(xmax, im_w), State.randint(ymax, im_h)
    im_crop = im[begin_y: end_y+1, begin_x: end_x+1, :]
    pts_crop = list_pts - np.array([begin_x, begin_y])
    return im_crop, pts_crop
    
