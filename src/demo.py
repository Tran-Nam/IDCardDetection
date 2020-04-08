import numpy as np 
import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2 
import sys 
import time
sys.path.append('..')
from data.data_utils import resize
import tensorflow as tf 
from utils.pafprocess import paf_to_idcard
import os
import argparse


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

ap = argparse.ArgumentParser()
ap.add_argument('--model_path', required=True, help='path to checkpoint')
ap.add_argument('--save_heatmap', action='store_true', help='save heatmap or not')
ap.add_argument('--save_paf', action='store_true', help='save paf or not')
ap.add_argument('--image_dir', required=True, help='path to image')
ap.add_argument('--output_dir', required=True, help='output save results')
args = ap.parse_args()

model_path = args.model_path

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
    hm = hm[0]
    paf = paf[0]
    off = off[0]
    results = paf_to_idcard(hm, paf, off, ratio=4)
    for i in range(len(results['pts'])):
        results['pts'][i][:8][::2] -= pad_x 
        results['pts'][i][:8][1::2] -= pad_y
        results['pts'][i][:8] /= ratio
    hm = sigmoid(hm)
    hm_show = show_hm(hm)
    paf_show = show_paf(paf)
    return results, hm_show, paf_show

im_dir = args.image_dir
out_dir = args.output_dir
save_heatmap = args.save_heatmap
save_paf = args.save_paf
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

im_paths = os.listdir(im_dir)
ext = ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']
im_paths = [path for path in im_paths if path.split('.')[-1] in ext]
for path in im_paths:
    im = cv2.imread(os.path.join(im_dir, path))
    results, hm_show, paf_show = inference_single_image(sess, im)
    idcards = results['pts']
    missing_corners = results['missing_corners']
    if save_heatmap:
        cv2.imwrite(os.path.join(out_dir, path.split('.')[0]+'_hm.png'), hm_show)
    if save_paf:
        cv2.imwrite(os.path.join(out_dir, path.split('.')[0]+'_paf.png'), paf_show)
    
    for (idcard, missing_idx) in zip(idcards, missing_corners):
        pts = idcard[:8].reshape(4, 2).astype(np.int)
        score = idcard[-1]
        if len(missing_idx)==0:
            cv2.polylines(im, [pts.reshape(-1, 4, 2)], 1, (0, 0, 255))
        for j in range(4):
            if j in missing_idx:
                continue
            pt = pts[j]
            cv2.circle(im, tuple(pt), 20, COLOR_MAP[j], -1)
            
    cv2.imwrite(os.path.join(out_dir, path), im)
