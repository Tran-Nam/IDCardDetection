import os
import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import pickle
from scipy.ndimage import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure
import sys
sys.path.append('..')
import config

NUM_JOINTS = config.NUM_JOINTS
NUM_LIMBS = config.NUM_LIMBS
LIMBS = config.LIMBS
PAF_XY_COORDS_PER_LIMB = config.PAF_XY_COORDS_PER_LIMB

# sample_path = '../../data/sample.pkl'
# with open(sample_path, 'rb') as f:
#     sample = pickle.load(f)
# heatmap = sample['heatmap']
# paf = sample['paf']
# print(heatmap.shape, paf.shape)

def imshow(im):
    if len(im.shape)==3:
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(im, cmap='gray')
    plt.show()

def find_peaks(im, thresh):
    """
    find local maximum and has value greater than threshold
    """
    peaks_binary = (maximum_filter(im, footprint=generate_binary_structure(2, 1))==im) * (im>thresh)
    peaks_idx = np.array(np.nonzero(peaks_binary)[::-1]).T ## reverve to get [x, y] coordinate
    return peaks_idx

def find_joints(heatmap, thresh):
    """
    find info of each joint type in heatmap
    each joint type is a nx4 ndarray
    each joint is stored info [x, y, score, id_peak]
    :x, y - coordinate
    :score - value in heatmap 
    :id_peak - id of peak
    :return list of joints, each element in list is nx4 array store info of joint correspond joint type
    """
    joints_per_type = []
    joints_id = 0
    for jid in range(NUM_JOINTS):
        joint_heatmap = heatmap[:, :, jid]
        peaks_coord = find_peaks(joint_heatmap, thresh)
        peaks = np.zeros((len(peaks_coord), 4)) ### x, y, score, id_peak
        for i, peak in enumerate(peaks_coord):
            x, y = peak
            peak_score = joint_heatmap[y, x]
            peaks[i, :] = (x, y, peak_score, joints_id)
            joints_id += 1
        joints_per_type.append(peaks)
    return joints_per_type

def find_connected_joints(paf, joints_list_per_type, num_inter_points=10):
    """
    find connected joint - limb candidate from paf and joints list
    :paf - paf output from network
    :joints_list_per_type - list of joint per type limb, each peak has info x, y, score, id in limb type, id overall
    :return list of limbs, each limb has has 5 field: [joint_src_id, joint_dst_id, limb_score, joint_src_index, joint_dst_index]
    """
    connnected_limbs = []
    limb_inter_coord = np.empty((4, num_inter_points), dtype=np.intp) # store info of interpoints x,y coords paf, id of paf correspond limbs
    for limb_type in range(NUM_LIMBS):
        # print(limbs[limb_type])
        joints_src = joints_list_per_type[LIMBS[limb_type][0]]
        joints_dst = joints_list_per_type[LIMBS[limb_type][1]]
        if len(joints_src)==0 or len(joints_dst)==0: # if limb type has no candidate (peak)
            connnected_limbs.append([])
        else:
            connection_candidate = []
            # id of paf in output, 2 id: src point and dst point
            limb_inter_coord[2, :] = PAF_XY_COORDS_PER_LIMB[limb_type][0]
            limb_inter_coord[3, :] = PAF_XY_COORDS_PER_LIMB[limb_type][1]
            for i, joint_src in enumerate(joints_src):
                for j, joint_dst in enumerate(joints_dst):
                    limb_dir = joint_dst[:2] - joint_src[:2] # direction vector
                    limb_dist = np.sqrt(np.sum(limb_dir**2)) + 1e-8
                    limb_dir = limb_dir / limb_dist # normalize
                    limb_inter_coord[1, :] = np.round(np.linspace(
                        joint_src[0], joint_dst[0], num=num_inter_points
                    )) # interpoint x coord
                    limb_inter_coord[0, :] = np.round(np.linspace(
                        joint_src[1], joint_dst[1], num=num_inter_points
                    )) # interpoint y coord
                    inter_paf = paf[limb_inter_coord[0, :],
                                   limb_inter_coord[1, :],
                                   limb_inter_coord[2:4, :]].T

                    score_inter_pts = np.dot(inter_paf, limb_dir) # score 
                    score = score_inter_pts.mean()
                    score_penalizing_long_dist = score_inter_pts.mean()+min(0.5*paf.shape[0]/limb_dist-1, 0) # penalty long distance
                    criteria1 = (np.count_nonzero(score_inter_pts>0.6) > 0.5*num_inter_points) # no. score > threshold at least 40%
                    criteria2 = (score_penalizing_long_dist>-10)
                    if criteria1:# check criteria, criteria2???!!
                        connection_candidate.append(
                            [i, j, score,
                            score+joint_src[2]+joint_dst[2]]
                        )

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True) # sorted by score 
            connections = np.empty((0, 5), dtype=np.int)
            max_connections = min(len(joints_src), len(joints_dst))
            for potential_connection in connection_candidate:
                i, j, s = potential_connection[:3]
                # [joint_src_id, joint_dst_id, limb_score, joint_src_index, joint_dst_index]
                if i not in connections[:, 3] and j not in connections[:, 4]:
                    connections = np.vstack(
                        [connections, [joints_src[i][3], joints_dst[j][3], s, i, j]]
                    )
                    if len(connections) >= max_connections:
                        break

            connnected_limbs.append(connections)
    return connnected_limbs

def group_limbs_same_idcard(connected_limbs, joint_list):
    """
    group limbs to be same idcard
    :connected_limbs - list of connected limbs candidate
    :joins_list - list of found joints
    :return nx(n_joints+2)
        first n_joints: index of joint
        2nd last: score overall 
        last: no. joints found in idcard
    """
    idcard_to_joint_assoc = []
    for limb_type in range(NUM_LIMBS):
        joint_src_type, joint_dst_type = LIMBS[limb_type]
        for limb_info in connected_limbs[limb_type]:
            idcard_assoc_idx = []
            # find joints current in list
            for idcard, idcard_limbs in enumerate(idcard_to_joint_assoc):
                if idcard_limbs[joint_src_type]==limb_info[0] or \
                    idcard_limbs[joint_dst_type]==limb_info[1]: # if already in list
                    idcard_assoc_idx.append(idcard) # append index to check later
                    
            if len(idcard_assoc_idx)==1: # found 1 joints
                idcard_limbs = idcard_to_joint_assoc[idcard_assoc_idx[0]]
                if idcard_limbs[joint_dst_type]!=limb_info[1]: # add to list
                    idcard_limbs[joint_dst_type] = limb_info[1]
                    # update info: score, num joints in limbs
                    idcard_limbs[-1] += 1
                    idcard_limbs[-2] += joint_list[
                        limb_info[1].astype(int), 2
                    ] + limb_info[2]
                    
            elif len(idcard_assoc_idx)==2: # found 2 joints, merge them
                idcard1_limbs = idcard_to_joint_assoc[idcard_assoc_idx[0]]
                idcard2_limbs = idcard_to_joint_assoc[idcard_assoc_idx[1]]
                membership = ((idcard1_limbs>=0)&(idcard2_limbs>=0))[:-2]
                if not membership.any(): # if 2 idcard has no connect joint, merge them
                    idcard1_limbs[:-2] += (idcard2_limbs[:-2]+1) # update joint connected
                    idcard1_limbs[-2:] += idcard2_limbs[-2:] # add score
                    idcard1_limbs[-2] += limb_info[2] # overall score
                    idcard_to_joint_assoc.pop(idcard_assoc_idx[1])
                else: # same with len==1 above
                    idcard1_limbs[joint_dst_type] = limb_info[1]
                    idcard1_limbs[-1] += 1
                    idcard1_limbs[-2] += joint_list[limb_info[1].astype(int), 2] + limb_info[2]
            else: # if not in list, create new idcard
                row = -1*np.ones(NUM_LIMBS+2)
                row[joint_src_type] = limb_info[0]
                row[joint_dst_type] = limb_info[1]
                row[-1] = 2
                row[-2] = sum(joint_list[limb_info[:2].astype(int), 2]) + limb_info[2]
                idcard_to_joint_assoc.append(row)

    # delete idcard has very few limbs
    idcard_to_delete = []
    for idcard_id, idcard_info in enumerate(idcard_to_joint_assoc):
        if idcard_info[-1] < 2 or idcard_info[-2]/idcard_info[-1] < 0.2: # not enough 4 corner
            idcard_to_delete.append(idcard_id)     
    for index in idcard_to_delete[::-1]:
        idcard_to_joint_assoc.pop(index)
    return np.array(idcard_to_joint_assoc)

def paf_to_idcard(heatmap, paf, offset, ratio=4):
    joints_per_type = find_joints(heatmap, 0.5)
    joints_list = np.array([
        tuple(peak) + (joint_type, ) for joint_type, joint_peak in enumerate(joints_per_type) for peak in joint_peak
    ]) 
    connected_limbs = find_connected_joints(paf, joints_per_type)      
    idcard_to_joint_assoc = group_limbs_same_idcard(connected_limbs, joints_list)

    idcards = []
    for idcard_id, idcard_info in enumerate(idcard_to_joint_assoc):
        joints = idcard_info[: NUM_JOINTS]
        idcard_score = idcard_info[-2]
        idcard = np.zeros((9, ), dtype=np.float32)
        for i in range(NUM_JOINTS):
            joint = joints[i]
            if joint==-1: #not found joint
                continue     
            peak_coord = joints_list[int(joint)][:2]         
            x, y = peak_coord.astype('int')
            # addition offset
            peak_coord += offset[y, x, :]
            # idcard[2*i: 2*(i+1)] = peak_coord
            idcard[2*i: 2*(i+1)] = (peak_coord*ratio).astype('int')
        idcard[-1] = idcard_score
        idcards.append(idcard)
    return idcards


# offset = np.zeros((512, 512, 2))

# idcards = paf_to_idcard(heatmap, paf, offset, ratio=1)
    
# im = np.zeros((512, 512, 3), dtype=np.uint8)
# pts = idcards[0]
# # print(pts.shape)
# pts = pts[:8].reshape(4, 2)
# for pts in idcards:
#     pts = pts[:8].reshape(4, 2)
#     for pt in pts:
#         if pt[0] >= 0:
#             cv2.circle(im, tuple(pt), 10, (0, 0, 255), -1)
#         cv2.polylines(im, [pts.astype(np.int32).reshape(1, 4, 2)], True, (0, 255, 0))
# plt.figure(figsize=(10, 10))
# imshow(im)