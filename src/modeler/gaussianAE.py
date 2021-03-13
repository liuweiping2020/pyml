import tensorflow as tf

from common.datautils import DataUtils
from modeler.tfmodel import TFModel


class GaussianAutoencoderModel(TFModel):
    def __init__(self, n_input=None, n_hidden=None, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(),
                 scale=0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights
        self.optimize=optimizer

        # model
    def build(self):
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + self.scale * tf.random_normal((self.n_input,)),
                                                     self.weights['w1']),
                                           self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

        # cost
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = self.optimizer.minimize(self.cost)

    def add_placeholder(self):
        self.scale = tf.placeholder(tf.float32)
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        pass

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(DataUtils.xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

