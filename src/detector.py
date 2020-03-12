import numpy as np 
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2 
import sys 
import time
sys.path.append('..')
from data.data_utils import resize
import tensorflow as tf 
from utils.pafprocess import paf_to_idcard
import os

model_path = '../ckpt/82000.ckpt'

COLOR_MAP = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (0, 255, 255)
]

sess = tf.compat.v1.Session()
saver = tf.compat.v1.train.import_meta_graph(model_path+'.meta')
saver.restore(sess, model_path)
training = tf.get_default_graph().get_tensor_by_name('training:0')
image = tf.get_default_graph().get_tensor_by_name('input_image:0')
hm_node = tf.get_default_graph().get_tensor_by_name('heatmap/last_convconv_block/conv2d/BiasAdd:0')
paf_node = tf.get_default_graph().get_tensor_by_name('paf/last_convconv_block/conv2d/BiasAdd:0')
off_node = tf.get_default_graph().get_tensor_by_name('offset/last_convconv_block/conv2d/BiasAdd:0')

def inference_single_image(sess, im):
    im_h, im_w = im.shape[:2]
    ratio = 512 / max(im_h, im_w)
    pad_x = (512 - im_w*ratio)//2
    pad_y = (512 - im_h*ratio)//2
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
    hm = np.squeeze(hm, axis=0)
    paf = np.squeeze(paf, axis=0)
    off = np.squeeze(off, axis=0)
    idcards = paf_to_idcard(hm, paf, off, ratio=4)
    for i in range(len(idcards)):
        idcards[i][:8][::2] -= pad_x 
        idcards[i][:8][1::2] -= pad_y
        idcards[i][:8] /= ratio
    return idcards

im_dir = '/hdd/namdng/CenterNet/data/cmt/images/'
out_dir = '/home/namtp/Desktop/IDCardDetection/out'
#im_path = '/hdd/IdCard/data_0304/train/images/1759.png'
im_paths = os.listdir(im_dir)
for path in im_paths:
    im = cv2.imread(os.path.join(im_dir, path))
    idcards = inference_single_image(sess, im)
    for idcard in idcards:
        pts = idcard[:8].reshape(4, 2).astype(np.int)
        cv2.polylines(im, [pts.reshape(-1, 4, 2)], 1, (0, 0, 255))
        for j in range(4):
            pt = pts[j]
            cv2.circle(im, tuple(pt), 10, COLOR_MAP[j], -1)
            
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
