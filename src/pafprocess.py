import os
import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import pickle
from scipy.ndimage import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure

n_kps = 4
n_limbs = 4
limbs = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0]
]
sample_path = '../data/sample.pkl'
with open(sample_path, 'rb') as f:
    sample = pickle.load(f)
heatmap = sample['heatmap']
paf = sample['paf']
print(heatmap.shape, paf.shape)

def imshow(im):
    if len(im.shape)==3:
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(im, cmap='gray')
    plt.show()

def find_peaks(im, thresh):
    peaks_binary = (maximum_filter(im, footprint=generate_binary_structure(2, 1))==im) * (im>thresh)
#     peaks_binary = (maximum_filter(im, size=8)==im) * (im>thresh)
    peaks_idx = np.array(np.nonzero(peaks_binary)[::-1]).T ## reverve to get [x, y] coordinate
    return peaks_idx

def find_kps(heatmap, thresh):
    kps_per_type = []
    kps_id = 0
    for kp in range(n_kps):
        kps_heatmap = heatmap[:, :, kp]
        #kps_heatmap = gaussian_filter(kps_heatmap, sigma=1)
        peaks_coord = find_peaks(kps_heatmap, thresh)
        peaks = np.zeros((len(peaks_coord), 4)) ### x, y, score, id_peak
        for i, peak in enumerate(peaks_coord):
            x, y = peak
            peak_score = kps_heatmap[y, x]
            peaks[i, :] = (x, y, peak_score, kps_id)
            kps_id += 1
        kps_per_type.append(peaks)
    return kps_per_type

joints_per_type = find_kps(heatmap, 0.5)

paf_xy_coords_per_limb = np.arange(8).reshape(4, 2)

def find_connected_joints(paf, joints_list_per_type, num_inter_points=10):
    connnected_limbs = []
    limb_inter_coord = np.empty((4, num_inter_points), dtype=np.intp)
    for limb_type in range(n_limbs):
        print(limbs[limb_type])
        joints_src = joints_list_per_type[limbs[limb_type][0]]
        joints_dst = joints_list_per_type[limbs[limb_type][1]]
        if len(joints_src)==0 or len(joints_dst)==0:
            connnected_limbs.append([])
        else:
            connection_candidate = []
            limb_inter_coord[2, :] = paf_xy_coords_per_limb[limb_type][0]
            limb_inter_coord[3, :] = paf_xy_coords_per_limb[limb_type][1]
            for i, joint_src in enumerate(joints_src):
                for j, joint_dst in enumerate(joints_dst):
#                     print(joint_src, joint_dst)
#                     input()
                    limb_dir = joint_dst[:2] - joint_src[:2]
                    limb_dist = np.sqrt(np.sum(limb_dir**2)) + 1e-8
                    limb_dir = limb_dir/limb_dist #normalize
#                     print(limb_dir)
#                     input()
                    
#                     print(joints_src)
                    limb_inter_coord[1, :] = np.round(np.linspace(
                        joint_src[0], joint_dst[0], num=num_inter_points
                    ))
                    limb_inter_coord[0, :] = np.round(np.linspace(
                        joint_src[1], joint_dst[1], num=num_inter_points
                    ))
#                     print(limb_inter_coord)
#                     input()
                    
                    inter_paf = paf[limb_inter_coord[0, :],
                                   limb_inter_coord[1, :],
                                   limb_inter_coord[2:4, :]].T
#                     print(inter_paf)
#                     input()
                    score_inter_pts = np.dot(inter_paf, limb_dir)
#                     print(score_inter_pts)
                    score_penalizing_long_dist = score_inter_pts.mean()+min(0.5*paf.shape[0]/limb_dist-1, 0)
#                     print(score_penalizing_long_dist)
                    criteria1 = (np.count_nonzero(
                        score_inter_pts>0.2
                    ) > 0.3*num_inter_points)
                    criteria2 = (score_penalizing_long_dist>-10)
                    if criteria1:
#                         print('ADD')
#                         input()
                        connection_candidate.append(
                            [i, j, score_penalizing_long_dist,
                            score_penalizing_long_dist+joint_src[2]+joint_dst[2]]
                        )
#                     if len(connection_candidate)!=0:
#                         print(connection_candidate)
#                     input()
            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connections = np.empty((0, 5), dtype=np.int)
            max_connections = min(len(joints_src), len(joints_dst))
#             print(connection_candidate)
#             input()
            for potential_connection in connection_candidate:
                i, j, s = potential_connection[:3]
                # [joint_src_id, joint_dst_id, limb_score_penalizing_long_dist, joint_src_index, joint_dst_index]
                if i not in connections[:, 3] and j not in connections[:, 4]:
#                     print(joints_src[i][:2], joints_dst[j][:2])
                    connections = np.vstack(
                        [connections, [joints_src[i][3], joints_dst[j][3], s, i, j]]
                    )
#                     print(joints_src[i], joints_dst[j])
                    if len(connections) >= max_connections:
                        break
#                 else:
#                     print('already in list')
#             print(connections)
#             input()
            connnected_limbs.append(connections)
    return connnected_limbs

connected_limbs = find_connected_joints(paf, joints_per_type)                    

joint_list = np.array([
    tuple(peak) + (joint_type, ) for joint_type, joint_peak in enumerate(joints_per_type) for peak in joint_peak
])

def group_limbs_same_person(connected_limbs, joint_list):
    person_to_joint_assoc = []
    for limb_type in range(n_limbs):
        joint_src_type, joint_dst_type = limbs[limb_type]
#         print(limbs[limb_type])
#         input()
        for limb_info in connected_limbs[limb_type]:
#             print(limb_info)
#             input()
            person_assoc_idx = []
            for person, person_limbs in enumerate(person_to_joint_assoc):
                
                if person_limbs[joint_src_type]==limb_info[0] or \
                    person_limbs[joint_dst_type]==limb_info[1]:
#                     print('ABC')
#                     print(person_limbs)
#                     print(limb_info)
                    person_assoc_idx.append(person)
                    
            if len(person_assoc_idx)==1:
                print('1111111111')
                person_limbs = person_to_joint_assoc[person_assoc_idx[0]]
                if person_limbs[joint_dst_type]!=limb_info[1]:
                    person_limbs[joint_dst_type] = limb_info[1]
                    person_limbs[-1] += 1
                    person_limbs[-2] += joint_list[
                        limb_info[1].astype(int), 2
                    ] + limb_info[2]
                    
            elif len(person_assoc_idx)==2:
                print('222222222')
                person1_limbs = person_to_joint_assoc[person_assoc_idx[0]]
                person2_limbs = person_to_joint_assoc[person_assoc_idx[1]]
#                 print(person1_limbs)
#                 print(person2_limbs)
                membership = ((person1_limbs>=0)&(person2_limbs>=0))[:-2]
                if not membership.any():
                    person1_limbs[:-2] += (person2_limbs[:-2]+1)
                    person1_limbs[-2:] += person2_limbs[-2:]
                    person1_limbs[-2] += limb_info[2]
                    person_to_joint_assoc.pop(person_assoc_idx[1])
                else:
                    person1_limbs[joint_dst_type] = limb_info[1]
                    person1_limbs[-1] += 1
                    person1_limbs[-2] += joint_list[limb_info[1].astype(int), 2] + limb_info[2]
            else:
                print('NEWWW')
                row = -1*np.ones(n_limbs+2)
                row[joint_src_type] = limb_info[0]
                row[joint_dst_type] = limb_info[1]
                row[-1] = 2
                row[-2] = sum(joint_list[limb_info[:2].astype(int), 2]) + limb_info[2]
                person_to_joint_assoc.append(row)
#             print('HERE')
#             print(person_to_joint_assoc)
#             input()
    people_to_delete = []
    for person_id, person_info in enumerate(person_to_joint_assoc):
        if person_info[-1] < 2 or person_info[-2]/person_info[-1] < 0.2:
            people_to_delete.append(person_id)     
    for index in people_to_delete[::-1]:
        person_to_joint_assoc.pop(index)
    return np.array(person_to_joint_assoc)
joint_list = np.array([
    tuple(peak) + (joint_type, ) for joint_type, joint_peak in enumerate(joints_per_type) for peak in joint_peak
])
# print(joints_per_type)
# print(joint_list)
# print(connected_limbs)
person_to_joint = group_limbs_same_person(connected_limbs, joint_list)

persons = []
for person_id, person_info in enumerate(person_to_joint):
    kps = person_info[:n_kps]
    person = np.empty((4, 2), dtype=np.float32)
    for i in range(n_kps):
        kp = kps[i]
#         if kp==-1:
#             continue
        if kp!=-1:
#             print(kp)
#             print(joint_list[int(kp)])
#             input()
            peak_coord = joint_list[int(kp)][:2]
            person[i] = peak_coord[:2]
        else:
            person[i] = [-1, -1]
    print(person)
#     person = np.array(person).reshape(n_kps, 2)
    persons.append(person)
    
im = np.zeros((512, 512, 3), dtype=np.uint8)
pts = persons[0]
for pts in persons:
    for pt in pts:
        if pt[0] >= 0:
            cv2.circle(im, tuple(pt), 10, (0, 0, 255), -1)
        cv2.polylines(im, [pts.astype(np.int32).reshape(1, 4, 2)], True, (0, 255, 0))
plt.figure(figsize=(10, 10))
imshow(im)