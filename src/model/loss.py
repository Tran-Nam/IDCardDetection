import numpy as np 
import tensorflow as tf
from tensorflow.python.ops import array_ops
import sys
# sys.path.append('../..')
# from data.processing import gen_heatmap_groundtruth


def smooth_l1_loss(pred, gt, sigma=1):
    diff = tf.abs(tf.subtract(pred, gt))
    smooth_l1_sign = tf.cast(tf.less(diff, sigma), tf.float32) # 0, 1
    smooth_l1_case1 = tf.multiply(diff, diff) / 2.0
    smooth_l1_case2 = tf.subtract(diff, 0.5*sigma) * sigma
    return tf.add(tf.multiply(smooth_l1_sign, smooth_l1_case1), \
        tf.multiply(tf.abs(tf.subtract(smooth_l1_sign, 1.0)), smooth_l1_case2))

def offset_loss(pred, gt, mask):
    n_corner = tf.reduce_sum(mask)  
    mask = tf.stack((mask, mask), axis=-1) # mask for 2 axis in offset map
    l1_loss = smooth_l1_loss(pred, gt)
    l1_loss /= (n_corner + tf.convert_to_tensor(1e-6, dtype=tf.float32))
    # print(l1_loss.get_shape())
    return tf.reduce_sum(tf.multiply(l1_loss, mask))
    # return tf.reduce_mean(tf.multiply(l1_loss, mask))

def focal_loss(pred, gt, alpha=2, beta=4):
    #n_sample_per_batch = gt.get_shape().as_list()[0]
    pred = tf.nn.sigmoid(pred)####### important
    zeros = array_ops.zeros_like(pred, dtype=pred.dtype)
    ones = array_ops.ones_like(pred, dtype=pred.dtype)
    gt = tf.cast(gt, dtype=pred.dtype)
    pos_p_sub = tf.where(tf.equal(gt, 1), ones-pred, zeros)
    neg_p_sub = tf.where(tf.less(gt, 1), pred, zeros)
    reduce_penalty = ones-gt
    per_entry_loss = -(pos_p_sub**alpha * tf.log(tf.clip_by_value(pred, 1e-8, 1.0)) \
        + reduce_penalty**beta * neg_p_sub**alpha * tf.log(tf.clip_by_value(1-pred, 1e-8, 1.0)))
    # print(per_entry_loss.get_shape())
    return tf.reduce_sum(per_entry_loss)# / tf.convert_to_tensor(n_sample_per_batch)
    # return tf.reduce_mean(per_entry_loss)

def mse_loss(pred, gt):
    #n_sample_per_batch = gt.get_shape().as_list()[0]
    return tf.reduce_sum(tf.keras.losses.MSE(pred, gt))# / tf.convert_to_tensor(n_sample_per_batch)
    # return tf.reduce_mean(tf.keras.losses.MSE(pred, gt))

def loss(pred, gt):
    hm_gt = gt[:, :, :, 0:4]
    paf_gt = gt[:, :, :, 4:12]
    off_gt = gt[:, :, :, 12:14]
    mask_gt = gt[:, :, :, 14:15]
    hm_pred = pred[:, :, :, 0:4]
    paf_pred = pred[:, :, :, 4:12]
    off_pred = pred[:, :, :, 12:14]
    f_loss = focal_loss(hm_pred, hm_gt)
    p_loss = mse_loss(paf_pred, paf_gt)
    o_loss = offset_loss(off_pred, off_gt, mask_gt)
    return f_loss + p_loss + o_loss

if __name__=='__main__':
    shape = (10, 128, 128, 8)
    gt = tf.zeros(shape)
    a = tf.reduce_mean(gt)
    b = tf.reduce_sum(gt)
    pred = tf.zeros(shape)
    gt_offset = tf.ones(shape=(10, 128, 128, 2))
    pred_offset = tf.zeros(shape=(10, 128, 128, 2))
    mask = tf.ones(shape=shape[:3])
    loss_mse = mse_loss(pred, gt)
    loss_focal = focal_loss(pred, gt)
    loss_offset = offset_loss(pred_offset, gt_offset, mask)
    with tf.Session() as sess:
        le, lf, lo, a, b = sess.run([loss_mse, loss_focal, loss_offset, a, b])
        print(le, lf, lo)
        # print(a, b)