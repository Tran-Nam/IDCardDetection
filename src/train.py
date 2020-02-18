import tensorflow as tf 
import numpy as np 
import datetime
print(tf.__version__)

from model.network import net
import config

class Trainer():
    def __init__(self):
        self.model = net()
        self.lr = config.LEARNING_RATE
        self.pretrained = config.PRETRAINED
        self.pretrained_path = config.PRETRAINED_PATH
        self.model_dir = config.MODEL_DIR
    
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

    def train(self):
        



if __name__=='__main__':
    shape = (10, 256, 256, 3)
    im = tf.ones(shape)
    
    training = tf.placeholder(tf.bool, name='training')
    
    a = net(im, is_training=training)
    post_fix = datetime.date.today().strftime("%d-%m-%Y")
    writer = tf.summary.FileWriter('./logs/{}'.format(post_fix), tf.get_default_graph())
    sess = tf.Session()
    sess.run(tf.global_variables_initializer()) #### init after build graph
    # nodes = sess.graph.as_graph_def().node
    # [print(node.name) for node in nodes]
    # print(len(nodes))
    out = sess.run(a, feed_dict={training: False})
    for i, j in out.items():
        print(i, j.shape)