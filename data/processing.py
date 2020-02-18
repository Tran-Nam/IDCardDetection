import numpy as np 
import cv2 
import data_utils
import matplotlib.pyplot as plt

skeleton = [
    [0, 1],
    [1, 2], 
    [2, 3], 
    [3, 0]
]

def gen_paf(im, line, sigma=3, eps=1e-6):
    """
    generate paf for 2 keypoints
    :param im: image
    :param line: 2 keypoints, format [kp1, kp2] 
    :param sigma: "width" of paf
    :param eps: avoid number compare error
    """
    im_h, im_w = im.shape[:2]
    base_pt, head_pt = line
    direction_vt = data_utils.find_direction_vector(line)
    perpendicular_vt = data_utils.find_perpendicular_vector(direction_vt)
    length_part = data_utils.cal_length_line(line)

    horizontal_axis = np.arange(im_w)
    vetical_axis = np.arange(im_h)
    x_coor, y_coor = np.meshgrid(horizontal_axis, vetical_axis)

    x_vt = x_coor - base_pt[0]
    y_vt = y_coor - base_pt[1]
    xy_vt = np.stack((x_vt, y_vt), axis=2)

    perpendicular_mask = np.sum(xy_vt*direction_vt, axis=2)
    perpendicular_mask = np.where(np.logical_and(
        perpendicular_mask >= -eps,
        perpendicular_mask <= length_part+eps
    ), 1, 0)

    direction_mask = np.sum(xy_vt*perpendicular_vt, axis=2)
    direction_mask = np.where(abs(direction_mask) < sigma+eps, 1, 0)

    full_direction = np.full((im_h, im_w, 2), direction_vt)

    total_mask = np.expand_dims(perpendicular_mask, axis=2)*\
        np.expand_dims(direction_mask, axis=2)
    # print(total_mask.shape)
    total_mask = np.squeeze(total_mask, axis=-1)

    paf = np.expand_dims(perpendicular_mask, axis=2)*\
        np.expand_dims(direction_mask, axis=2)*\
        full_direction
    return total_mask, paf

def gen_paf_groundtruth(im, list_pts):
    """
    generate groundtruth for image with paf mat of all limb
    :param im: image
    :param list_pts: list of keypoints of object in image
        format: [[tl1, tr1, br1, bl1]
            [...]]
    """
    im_h, im_w = im.shape[:2]
    paf_mat = np.zeros((im_h, im_w, 2*len(skeleton)))
    paf_mask = np.zeros((im_h, im_w, len(skeleton)))
    for pts in list_pts:
        for i in range(len(skeleton)):
            parts = skeleton[i]
            line = [pts[parts[0]], pts[parts[1]]]
            mask, paf = gen_paf(im, line)
            paf_mat[:, :, i*2: 2*(i+1)] += paf 
            paf_mask[:, :, i] += mask
    paf_mask = np.repeat(paf_mask, 2, axis=-1)
    paf_mat = np.divide(paf_mat, paf_mask, out=np.zeros_like(paf_mat), where=paf_mask!=0) 
    # https://stackoverflow.com/questions/26248654/how-to-return-0-with-divide-by-zero/37977222#37977222
    return paf_mat

if __name__=='__main__':
    im = np.zeros((512, 512, 3), dtype=np.uint8)
    list_pts = [
        [[50, 50],
        [400, 60],
        [450, 300],
        [30, 250]],

        [[0, 300],
        [400, 150],
        [400, 500],
        [20, 500]],

        [[300, 300],
        [450, 310],
        [400, 470],
        [270, 500]],

        [[10, 200],
        [200, 250],
        [250, 400],
        [50, 350]]     
    ]
    """
    # pts = [
    #     [50, 50],
    #     [400, 60],
    #     [450, 300],
    #     [30, 250]
    # ]
    # line = [
    #     [50, 50],
    #     [500, 100]
    # ]
    for pts in list_pts:
        pts = np.array(pts).reshape(4, 1, 2)
        cv2.polylines(im, [pts], 1, (255, 255, 255), 5)
        # for pt in pts:
        #     cv2.circle(im, tuple(pt), 3, (255, 255, 255), -1)
    cv2.imwrite('a.png', im)

    im_h, im_w = im.shape[:2]
    paf_mat = np.zeros((im_h, im_w, 2*len(skeleton)))
    paf_mask = np.zeros((im_h, im_w, len(skeleton)))

    for pts in list_pts:
        for i in range(len(skeleton)):
            part = skeleton[i]
            line = [pts[part[0]], pts[part[1]]]
            # print(line)
            mask, paf = gen_paf(im, line)
            # print(mask.shape, paf.shape)
            # print(i)
            paf_mat[:, :, i*2: 2*(i+1)] += paf 
            paf_mask[:, :, i] += mask
            # print(np.unique(paf[:, :, 0]), np.unique(paf[:, :, 1]))
            # sum_paf = np.sum(paf**2, axis=2)
            # print(np.unique(sum_paf))
            # sum_paf = (sum_paf*255).astype(np.uint8)
            # cv2.imwrite('{}.png'.format(part[0]), sum_paf)
    paf_mask = np.repeat(paf_mask, 2, axis=-1)
    print(paf_mask.shape)
    print(np.max(paf_mask[:, :, 0] - paf_mask[:, :, 1]))
    average_mask = np.sum(paf_mask, axis=(0, 1)).reshape(1, 1, 2*len(skeleton))
    print(average_mask.shape)
    print(average_mask)
    # paf_mat = paf_mat / paf_mask[paf_mask!=0]
    paf_mat = np.divide(paf_mat, paf_mask, out=np.zeros_like(paf_mat), where=paf_mask!=0) 
    #refer https://stackoverflow.com/questions/26248654/how-to-return-0-with-divide-by-zero/37977222#37977222
    print(paf_mat.shape)
    """

    paf_mat = gen_paf_groundtruth(im, list_pts)

    """
    #@@@ visual check
    # fig = plt.figure(dpi=500)
    rx_ = paf_mat[:, :, 2]
    ry_ = paf_mat[:, :, 3]
    x_show = rx_ / (np.sqrt(rx_**2 + ry_**2)+1e-6)
    y_show = ry_ / (np.sqrt(rx_**2 + ry_**2)+1e-6)
    plt.quiver(x_show, y_show)
    plt.show()
    # plt.savefig('d.png')
    """

    
    for i in range(len(skeleton)):
        sum_paf = np.sum(paf_mat[:, :, 2*i: 2*(i+1)]**2, axis=2)
        sum_paf = (sum_paf*255).astype(np.uint8)
        print(np.unique(sum_paf))
        cv2.imwrite('{}.png'.format(i), sum_paf)


    # paf = gen_paf(im, line)
    # # paf_x = paf[:, :, 0]
    # # paf_y = paf[:, :, 1]
    # sum_paf = np.sum(paf**2, axis=2)
    # sum_paf = (sum_paf*255).astype(np.uint8)
    # print(np.unique(sum_paf))
    # cv2.imwrite('b.png', sum_paf)
