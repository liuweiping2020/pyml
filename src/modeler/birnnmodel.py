from modeler.tfmodel import TFModel
import tensorflow as tf


class BiRNNModel(TFModel):
    def __init__(self):
        self.learning_rate = 0.01
        self.batch_size = 128
        self.display_step = 10
        self.n_input = 28  # MNIST data input (img shape: 28*28)
        self.n_steps = 28  # timesteps
        self.n_hidden = 256  # hidden layer num of features
        self.n_classes = 10  # MNIST total classes (0-9 digits)

        pass

    def add_placeholder(self):
        # tf Graph input
        self.x = tf.placeholder("float", [None, self.n_steps, self.n_input])
        self.y = tf.placeholder("float", [None, self.n_classes])
        pass

    def build(self):
        # Define weights
        weights = {
            # Hidden layer weights => 2*n_hidden because of foward + backward cells
            'out': tf.Variable(tf.random_normal([2 * self.n_hidden, self.n_classes]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }

        pred = self.BiRNN(self.x, weights, biases)

        # Define loss and optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        pass

    def BiRNN(self, x, weights, biases):
        x = tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [-1, self.n_input])
        x = tf.split(x, self.n_steps)

        # Define lstm cells with tensorflow
        # Forward direction cell
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)
        # Backward direction cell
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)

        # Get lstm cell output
        outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                                dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']
