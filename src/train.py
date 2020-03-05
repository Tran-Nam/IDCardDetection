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
from dataset.dataloader import input_fn

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
        self.num_epochs = 500
        config_gpu = tf.ConfigProto()
        config_gpu.gpu_options.allow_growth = True
        self.session = tf.Session(config=config_gpu)
        #self.saver = tf.train.Saver(max_to_keep=3)
        num_sample_train = 474
        num_sample_val = 53
        self.steps_per_epoch_train = num_sample_train // self.batch_size
        self.steps_per_epoch_val = num_sample_val // self.batch_size

        # data param
        self.train_data_path = config.TRAIN_DATA_PATH
        self.val_data_path = config.VAL_DATA_PATH
        
    
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

    def train(self, is_training):
        training = tf.placeholder_with_default(False, shape=(), name='training')
        with tf.variable_scope('train_data_pipeline'):
            batch, iterator_init_op = input_fn(filenames=self.train_data_path, is_training=is_training)
        image = batch['image']
        heatmap = batch['heatmap']
        offset = batch['offset']
        paf = batch['paf']
        mask = batch['mask']

        output = self.model(inputs=image, is_training=training)
        heatmap_pred = output['heatmap']
        paf_pred = output['affi']
        offset_pred = output['offset']

        with tf.variable_scope('loss'):
            f_loss = focal_loss(heatmap_pred, heatmap)
            o_loss = offset_loss(offset_pred, offset, mask)
            p_loss = mse_loss(paf_pred, paf)
            loss = f_loss + o_loss + p_loss
            loss = tf.cast(loss, dtype=tf.float64)
            
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        

        with tf.variable_scope('optimizer'):
            global_step = tf.Variable(0, trainable=False, name='global_step')
            lr = tf.train.exponential_decay(self.lr, global_step, self.decay_step, self.decay_rate, staircase=True, name= 'learning_rate')
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss, global_step)
        #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        logdir = os.path.join('graphs', datetime.datetime.now().strftime('%m-%d-%Y'))
        writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
        saver = tf.train.Saver(max_to_keep=3)
        init = tf.global_variables_initializer() 

        tf.summary.scalar('focal_loss', f_loss)
        tf.summary.scalar('offset_loss', o_loss)
        tf.summary.scalar('paf_loss', p_loss)
        tf.summary.scalar('total_loss', loss)
        tf.summary.scalar('lr', lr)
        tf.summary.image('input', image, max_outputs=5)
        #tf.summary.image('top_left_gt', heatmap[:, :, :, 0:1], max_outputs=4)
        joints_gt = tf.reduce_sum(heatmap, axis=-1)
        tf.summary.image('joints_gt', tf.expand_dims(joints_gt, -1), max_outputs=4)
        heatmap_show = tf.nn.sigmoid(heatmap_pred)
        #tf.summary.image('top_left', heatmap_show[:, :, :, 0:1], max_outputs=4)
        joints_heatmap = tf.reduce_sum(heatmap_show, axis=-1)
        tf.summary.image('joints_pred', tf.expand_dims(joints_heatmap, -1), max_outputs=4)
        
        b, w, h, _ = paf_pred.get_shape().as_list() # bxwxhx8
        for i in range(4):
            limb_pred = paf_pred[:, :, :, 2*i:2*(i+1)]
            limb_pred = tf.square(limb_pred)
            limb_pred = tf.reduce_sum(limb_pred, axis=-1)
            tf.summary.image('limb_{}'.format(i), tf.expand_dims(limb_pred, -1), max_outputs=4)
        """
        print(b, w, h, _)
        paf_heatmap = tf.reshape(paf_pred, [b, w, h, 4, 2])
        paf_heatmap = tf.square(paf_heatmap)
        paf_heatmap = tf.reduce_sum(paf_heatmap, axis=-1)
        paf_heatmap = tf.reduce_sum(paf_heatmap, axis=-2)
        tf.summary.image('paf_pred', paf_heatmap, max_outputs=4)
        """
        
        merge = tf.summary.merge_all()

        # sess = tf.Session()
        # sess.run(init)
        # sess.run(iterator_init_op)
        self.session.run(init)
        if self.pretrained:
            if self.load_ckpt(saver, self.session, self.pretrained_path):
                print('[*] Load SUCCESS!')
            else:
                print('[*] Load FAIL ...')
        
        
        for epoch in range(1, self.num_epochs+1):
            print('Epoch: {}/{}'.format(epoch, self.num_epochs))
            self.session.run(iterator_init_op)
            for step in range(self.steps_per_epoch_train):
                #self.session.run(update_ops)
                total_loss, hm_loss, of_loss, paf_loss, _ = self.session.run([loss, f_loss, o_loss, p_loss, train_op], feed_dict={training: True})
                if step!=0 and step%10==0:
                    #print('Step {}/{}'.format(step, self.steps_per_epoch_train))
                    print('Step {}/{} | loss {:.2f} | hm loss {:.2f} | o loss {:.2f} | paf loss {:.2f}'.format(step, self.steps_per_epoch_train, total_loss, hm_loss, of_loss, paf_loss))
 
            summary = self.session.run(merge, feed_dict={training: True})
            writer.add_summary(summary, epoch)
            if epoch > 100 and epoch%50==0:
                saver.save(self.session, os.path.join(self.model_dir, str(epoch)+'.ckpt'))
            
            # if epoch > 10:
            #     print('Evaluation ...')
            #     for step in range(self.steps_per_epoch_val):
            #         loss = self.session.run(loss)

        # for step in range(self.num_steps):
        #     print('HERE' + '='*50)
        #     input()
        #     loss_, _ = sess.run([loss, train_op], feed_dict={training: is_training})
        #     print('step %d, loss %g'%(step, loss_))

        #     if step%config.INTERVAL_SAVE==0 and step > 0:
        #         summary = sess.run(merge)
        #         writer.add_summary(summary, step)
        #         saver.save(sess, os.path.join(self.model_dir, str(step) + '.ckpt'))



if __name__=='__main__':
    T = Trainer()
    T.train(True)
