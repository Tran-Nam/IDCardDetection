import numpy as np 
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2 
import sys 
import time
sys.path.append('..')
from data.data_utils import resize
import tensorflow as tf 
from utils.pafprocess import paf_to_idcard
import os
import matplotlib.pyplot as plt
import seaborn as sns

model_path = '../checkpoint_4/90000.ckpt'

def sigmoid(x):
    return 1 / (1+np.exp(-x))
    
def show_hm(hm):
    hm = np.clip(np.sum(hm, axis=2), 0, 1)
    hm = (hm*255).astype(np.uint8)
    hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    return hm

def show_paf(paf):
    paf = paf.reshape(128, 128, 4, 2)
    paf = np.clip(np.sum(np.square(paf), axis=3), 0, 1)
    return show_hm(paf)
    
COLOR_MAP = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (0, 255, 255)
]

config_gpu = tf.ConfigProto()
config_gpu.gpu_options.allow_growth = True

sess = tf.compat.v1.Session(config=config_gpu)
saver = tf.compat.v1.train.import_meta_graph(model_path+'.meta')
saver.restore(sess, model_path)
training = tf.get_default_graph().get_tensor_by_name('training:0')
image = tf.get_default_graph().get_tensor_by_name('input_image:0')
hm_node = tf.get_default_graph().get_tensor_by_name('heatmap/last_convconv_block/conv2d/BiasAdd:0')
paf_node = tf.get_default_graph().get_tensor_by_name('paf/last_convconv_block/conv2d/BiasAdd:0')
off_node = tf.get_default_graph().get_tensor_by_name('offset/last_convconv_block/conv2d/BiasAdd:0')

def inference_single_image(sess, im):
    im_h, im_w = im.shape[:2]
    #print(im_h, im_w)
    ratio = 512 / max(im_h, im_w)
    pad_x = (512 - im_w*ratio)//2
    pad_y = (512 - im_h*ratio)//2
    #print(pad_x, pad_y)
    im_resize, _ = resize(im, [])
    im_expand = np.expand_dims(im_resize, axis=0)
    start = time.time()
    hm, paf, off = sess.run([
        hm_node, paf_node, off_node
    ], feed_dict={
        image: im_expand,
        training: False
    })
    print('SESSION: ', time.time() - start)
    hm = hm[0]
    # print(np.min(hm), np.max(hm))
    paf = paf[0]
    off = off[0]
    
    """
    hm = np.squeeze(hm, axis=0)
    paf = np.squeeze(paf, axis=0)
    off = np.squeeze(off, axis=0)
    """
    # idcards = paf_to_idcard(hm, paf, off, ratio=4)
    start = time.time()
    results = paf_to_idcard(hm, paf, off, ratio=4)
    print('POST PROCESS: ', time.time() - start)
    # idcards = results['pts']
    # missing_corners = results['missing_corners']
    for i in range(len(results['pts'])):
        results['pts'][i][:8][::2] -= pad_x 
        results['pts'][i][:8][1::2] -= pad_y
        results['pts'][i][:8] /= ratio
    hm = sigmoid(hm)
    hm_show = show_hm(hm)
    paf_show = show_paf(paf)
    return results, hm_show, paf_show

im_dir = '/home/namtp/Desktop/IDCardDetection/src/tmp/0324/cmnd_back'
out_dir = '/home/namtp/Desktop/IDCardDetection/src/tmp/test_0327/cmnd_back'
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)
#im_path = '/hdd/IdCard/data_0304/train/images/1759.png'
im_paths = os.listdir(im_dir)
im_paths = [path for path in im_paths if path.endswith('jpg') and 'CMT' not in path]
for path in im_paths:
    # if path!='0f9126aac39cde99d940.jpg':
    #     continue
    im = cv2.imread(os.path.join(im_dir, path))
    #for _ in range(10):
    results, hm_show, paf_show = inference_single_image(sess, im)
    idcards = results['pts']
    missing_corners = results['missing_corners']
    cv2.imwrite(os.path.join(out_dir, path.split('.')[0]+'_hm.png'), hm_show)
    cv2.imwrite(os.path.join(out_dir, path.split('.')[0]+'_paf.png'), paf_show)
    
    for (idcard, missing_idx) in zip(idcards, missing_corners):
        pts = idcard[:8].reshape(4, 2).astype(np.int)
        score = idcard[-1]
        # print(score)
        if len(missing_idx)==0:
            cv2.polylines(im, [pts.reshape(-1, 4, 2)], 1, (0, 0, 255))
        for j in range(4):
            if j in missing_idx:
                continue
            pt = pts[j]
            cv2.circle(im, tuple(pt), 20, COLOR_MAP[j], -1)
            
    cv2.imwrite(os.path.join(out_dir, path), im)
    """
    print(idcards)
    hm = sigmoid(hm)
    hm = np.clip(np.sum(hm, axis=2), 0, 1)
    print(np.min(hm), np.max(hm))
    hm = (hm*255).astype(np.uint8)
    print(hm.shape)
    hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    cv2.imwrite('hm.png', hm)

    print(paf.shape)
    for i in range(4):
        limb = paf[:, :, 2*i:2*(i+1)]
        print(np.min(limb), np.max(limb))
        limb = np.clip(np.sum(limb**2, axis=2), 0, 1)
        print(limb.shape)
        limb = (limb*255).astype(np.uint8)
        # limb = cv2.applyColorMap(limb, cv2.COLORMAP_JET)
        cv2.imwrite('limb_{}.png'.format(i), limb)
    """
