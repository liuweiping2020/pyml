import tensorflow as tf

from modeler.tfmodel import TFModel


class MLPModel(TFModel):
    def __init__(self):
        self.in_units = 784
        self.h1_units = 300
        pass

    def add_placeholder(self):
        self.x = tf.placeholder(tf.float32, [None, self.in_units])
        self.y_ = tf.placeholder(tf.float32, [None, 10])
        self.keep_prob = tf.placeholder(tf.float32)
        pass

    def build(self):
        W1 = tf.Variable(tf.truncated_normal([self.in_units, self.h1_units], stddev=0.1))
        b1 = tf.Variable(tf.zeros([self.h1_units]))
        W2 = tf.Variable(tf.zeros([self.h1_units, 10]))
        b2 = tf.Variable(tf.zeros([10]))

        hidden1 = tf.nn.relu(tf.matmul(self.x, W1) + b1)
        hidden1_drop = tf.nn.dropout(hidden1, self.keep_prob)
        y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)

        # Define loss and optimizer
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(y), reduction_indices=[1]))
        self.train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

        # Test trained model
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        pass
