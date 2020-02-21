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
        self.model = net
        self.lr = config.LEARNING_RATE
        self.pretrained = config.PRETRAINED
        self.pretrained_path = config.PRETRAINED_PATH
        self.model_dir = config.MODEL_DIR
        self.decay_rate = 0.95
        self.decay_step = 10000
        self.num_steps = 100
    
    def load_ckpt(self, saver, sess, model_path):
        """
        load pretrained weights
        """
        if os.path.exists(model_path + '.meta'):
            saver.restore(sess, model_path)
            print('Restore model from {}'.format(model_path))
            return True
        else:
            return False

    def train(self, is_training):
        training = tf.placeholder(tf.bool, name='training')
        with tf.variable_scope('data_pipeline'):
            batch, iterator_init_op = input_fn(filenames='../data/train.tfrecords', is_training=is_training)
        image = batch['image']
        heatmap = batch['heatmap']
        offset = batch['offset']
        paf = batch['paf']
        mask = batch['mask']
        print('aaa')
        print(image, heatmap, offset, mask, paf)
        input()

        output = self.model(inputs=image, is_training=training)
        heatmap_pred = output['heatmap']
        paf_pred = output['affi']
        offset_pred = output['offset']
        print('bbbbb')
        print(heatmap_pred, offset_pred, paf_pred)
        input()

        with tf.variable_scope('loss'):
            f_loss = focal_loss(heatmap_pred, heatmap)
            o_loss = offset_loss(offset_pred, offset, mask)
            p_loss = mse_loss(paf_pred, paf)
            loss = tf.add(f_loss, o_loss, p_loss)
            loss = tf.cast(loss, dtype=tf.float64)

        with tf.variable_scope('optimizer'):
            global_step = tf.Variable(0, trainable=False, name='global_step')
            lr = tf.train.exponential_decay(self.lr, global_step, self.decay_step, self.decay_rate, staircase=True, name= 'learning_rate')
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            train_op = optimizer.minimize(loss, global_step)

        writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
        saver = tf.train.Saver(max_to_keep=3)
        init = tf.global_variables_initializer() 

        tf.summary.scalar('focal_loss', f_loss)
        tf.summary.scalar('offset_loss', o_loss)
        tf.summary.scalar('paf_loss', p_loss)
        tf.summary.scalar('total_loss', loss)
        tf.summary.scalar('lr', lr)
        tf.summary.image('input', image, max_outputs=5)
        tf.summary.image('top_left_gt', heatmap[:, :, :, 0:1], max_outputs=5)
        tf.summary.image('top_left', heatmap_det[:, :, :, 0:1], max_outputs=5)
        merge = tf.summary.merge_all()

        sess = tf.Session()
        sess.run(init)
        sess.run(iterator_init_op)
        if self.pretrained:
            if self.load_ckpt(saver, sess, self.model_path):
                print('[*] Load SUCCESS!')
            else:
                print('[*] Load FAIL ...')
        
        for step in range(self.num_steps):
            loss_, _ = sess.run([loss, train_op], feed_dict={is_training: training})
            print('step %d, loss %g'%(step, loss_))

            if step%config.INTERVAL_SAVE==0 and step > 0:
                summary = sess.run(merge)
                writer.add_summary(summary, step)
                saver.save(sess, os.path.join(self.model_dir, str(step) + '.ckpt'))



if __name__=='__main__':
    T = Trainer()
    T.train(True)
    # shape = (10, 256, 256, 3)
    # im = tf.ones(shape)
    
    # training = tf.placeholder(tf.bool, name='training')
    
    # a = net(im, is_training=training)
    # post_fix = datetime.date.today().strftime("%d-%m-%Y")
    # writer = tf.summary.FileWriter('./logs/{}'.format(post_fix), tf.get_default_graph())
    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer()) #### init after build graph
    # # nodes = sess.graph.as_graph_def().node
    # # [print(node.name) for node in nodes]
    # # print(len(nodes))
    # out = sess.run(a, feed_dict={training: False})
    # for i, j in out.items():
    #     print(i, j.shape)