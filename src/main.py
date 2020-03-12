import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import tensorflow as tf 
import numpy as np 
import datetime
print(tf.__version__)

from model.network import net
from model.loss import focal_loss, offset_loss, mse_loss
import config
import sys
sys.path.append('..')
from dataset.custom_dataloader import Dataloader, augmentation

class Trainer():
    def __init__(self):
        # training param
        self.model = net
        self.lr = config.LEARNING_RATE
        self.batch_size = config.BATCH_SIZE
        self.pretrained = config.PRETRAINED
        self.pretrained_path = config.PRETRAINED_PATH
        self.model_dir = config.MODEL_DIR
        self.decay_rate = config.DECAY_RATE
        self.decay_step = config.DECAY_STEP
        self.num_steps = 100000
        self.interval_eval = 1000
        self.interval_save = 1000
        config_gpu = tf.ConfigProto()
        config_gpu.gpu_options.allow_growth = True
        self.session = tf.Session(config=config_gpu)
        self.train_dataloader = Dataloader(
            image_dir='/hdd/namtp/data/train/images',
            label_path='/hdd/namtp/data/train/label.csv',
            batch_size=self.batch_size,
            augmentation=augmentation(), 
            shuffle=True
        )
        self.val_dataloader = Dataloader(
            image_dir='/hdd/namtp/data/val/images',
            label_path='/hdd/namtp/data/val/label.csv',
            batch_size=self.batch_size,
            augmentation=None,
            shuffle=None
        )
        self._steps_per_eval = self.val_dataloader._step_per_epoch
        
           
    def load_ckpt(self, saver, session, model_path):
        """
        load pretrained weights
        """
        if os.path.exists(model_path + '.meta'): # check if exist checkpoint file -> restore
            saver.restore(session, model_path)
            print('Restore model from {}'.format(model_path))
            return True
        else:
            return False

    def main(self):
        training = tf.placeholder_with_default(False, shape=(), name='training')
        image = tf.placeholder(dtype=tf.float32, shape=[None, 512, 512, 3], name='input_image')
        groundtruth = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 15], name='input_groundtruth')
        hm_gt = groundtruth[:, :, :, 0:4]
        paf_gt = groundtruth[:, :, :, 4:12]
        off_gt = groundtruth[:, :, :, 12:14]
        mask_gt = groundtruth[:, :, :, 14] ### 2 dims

        pred = self.model(
            inputs=image,
            is_training=training 
        )
        #print(pred.shape)
        hm_pred = pred[:, :, :, 0:4]
        paf_pred = pred[:, :, :, 4:12]
        off_pred = pred[:, :, :, 12:14]
        #print(hm_pred.shape, paf_pred.shape, off_pred.shape)

        with tf.variable_scope('loss'):
            f_loss = focal_loss(hm_pred, hm_gt)
            p_loss = mse_loss(paf_pred, paf_gt)
            o_loss = offset_loss(off_pred, off_gt, mask_gt)
            loss = f_loss + p_loss + o_loss 
            # loss = tf.cast(loss, dtype=tf.float64) ??@@

        
        with tf.variable_scope('optimizer'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            global_step = tf.Variable(0, trainable=False)
            lr = tf.train.exponential_decay(self.lr, global_step, self.decay_step, self.decay_rate, staircase=True, name= 'learning_rate')
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss, global_step)
        init = tf.global_variables_initializer()
        logdir = os.path.join('graphs', datetime.datetime.now().strftime('%m-%d-%Y/%H-%M-%S'))
        writer_train = tf.summary.FileWriter(os.path.join(logdir, 'train'), tf.get_default_graph())
        writer_val = tf.summary.FileWriter(os.path.join(logdir, 'val'), tf.get_default_graph())

        sum_f_loss = tf.summary.scalar('hm loss', f_loss, collections=['train', 'val'])
        sum_p_loss = tf.summary.scalar('paf loss', p_loss, collections=['train', 'val'])
        sum_o_loss = tf.summary.scalar('off loss', o_loss, collections=['train', 'val'])
        sum_t_loss = tf.summary.scalar('total loss', loss, collections=['train', 'val'])
        sum_lr = tf.summary.scalar('learning rate', lr, collections=['train'])
        # train_sum = tf.summary.merge([sum_f_loss, sum_p_loss, sum_o_loss, sum_lr, sum_t_loss])
        
        tf.summary.image('image', image, max_outputs=1, collections=['train', 'image_val'])
        joints_gt = tf.reduce_sum(hm_gt, axis=-1)
        tf.summary.image('joints gt', tf.expand_dims(joints_gt, -1), max_outputs=1, collections=['train', 'image_val'])
        hm_pred_show = tf.nn.sigmoid(hm_pred)
        joints_pred = tf.reduce_sum(hm_pred_show, axis=-1)
        tf.summary.image('joints pred', tf.expand_dims(joints_pred, -1), max_outputs=1, collections=['train', 'image_val'])
        for limb_id in range(4):
            limb_gt = paf_gt[:, :, :, 2*limb_id: 2*(limb_id+1)]
            limb_gt = tf.square(limb_gt)
            limb_gt = tf.reduce_sum(limb_gt, axis=-1)
            tf.summary.image('limb gt {}'.format(limb_id), tf.expand_dims(limb_gt, -1), max_outputs=1, collections=['train', 'image_val'])
        for limb_id in range(4):
            limb_pred = paf_pred[:, :, :, 2*limb_id: 2*(limb_id+1)]
            limb_pred = tf.square(limb_pred)
            limb_pred = tf.reduce_sum(limb_pred, axis=-1)
            tf.summary.image('limb pred {}'.format(limb_id), tf.expand_dims(limb_pred, -1), max_outputs=1, collections=['train', 'image_val'])
        
        train_sum = tf.summary.merge_all(key='train')
        eval_sum = tf.summary.merge_all(key='val')
        image_val_sum = tf.summary.merge_all(key='image_val')
        saver = tf.train.Saver(max_to_keep=5)

        # print('-'*50)
        self.session.run(init)
        # print('='*50)
        if self.pretrained:
            if self.load_ckpt(saver, self.session, self.pretrained_path):
                print('[*] Load SUCCESS!')
            else:
                print('[*] Load FAIL ...')

        for step in range(1, self.num_steps+1):
            
            batch_im, batch_gt = self.train_dataloader.next_batch()
            #print(batch_im.shape, batch_gt.shape)
            to_loss, hm_loss, paf_loss, off_loss, summary_train, _ = self.session.run([
                loss, f_loss, p_loss, o_loss, train_sum, train_op
            ], feed_dict={
                image: batch_im,
                groundtruth: batch_gt,
                training: True 
            })
            writer_train.add_summary(summary_train, step)
            if step%10==0:
                print('Step {}/{}'.format(step, self.num_steps))
                print('Total loss: {:.2f} | hm loss: {:.2f} | paf loss: {:.2f} | offset loss: {:.2f}'.format(to_loss, hm_loss, paf_loss, off_loss))

            if step%self.interval_eval==0:
                print('EVAL ...')
                to_losses, hm_losses, paf_losses, off_losses = [], [], [], []
                ### EVAL
                for _ in range(self._steps_per_eval):
                    batch_im, batch_gt = self.val_dataloader.next_batch()
                    if _==0:
                        summary_image_val = self.session.run(image_val_sum, feed_dict={
                        image: batch_im,
                        groundtruth: batch_gt,
                        training: False 
                    })
                    #print('EVAL: ', batch_im.shape, batch_gt.shape)
                    to_loss, hm_loss, paf_loss, off_loss, summary_val = self.session.run([
                        loss, f_loss, p_loss, o_loss, eval_sum
                    ], feed_dict={
                        image: batch_im,
                        groundtruth: batch_gt,
                        training: False 
                    })
                    to_losses.append(to_loss)
                    hm_losses.append(hm_loss)
                    paf_losses.append(paf_loss)
                    off_losses.append(off_loss)
                print('LOSS: ', np.mean(hm_losses), np.mean(paf_losses))
                summary_val = self.session.run(eval_sum, feed_dict={
                    f_loss: np.mean(hm_losses),
                    o_loss: np.mean(off_losses),
                    p_loss: np.mean(paf_losses),
                    loss: np.mean(to_losses)
                })
                writer_val.add_summary(summary_image_val, step)
                writer_val.add_summary(summary_val, step)
                    #print(to_loss, hm_loss, paf_loss, off_loss)
            if step%self.interval_save==0:
                saver.save(self.session, os.path.join(self.model_dir, str(step)+'.ckpt'))
                #summary_val = self.session.run(eval_sum)
                #writer_val.add_summary(summary_val)
                
if __name__=='__main__':
    t = Trainer()
    t.main()
