import tensorflow as tf

from modeler.tfmodel import TFModel


class VGGModel(TFModel):
    def __init__(self):
        # super.__init__(VGGModel)
        self.batch_size = 32
        self.num_batches = 100
        self.image_size = 224
        pass

    def add_placeholder(self):
        self.input_op = tf.Variable(
            tf.random_normal([self.batch_size, self.image_size,
                              self.image_size, 3], dtype=tf.float32, stddev=1e-1))
        self.keep_prob = tf.placeholder(tf.float32)

    def build(self):
        self.predictions, self.softmax, self.fc8, self.parameters = \
            self.inference_op(self.input_op, self.keep_prob)
        pass

    def conv_op(self, input_op, name, kh, kw, n_out, dh, dw, p):
        n_in = input_op.get_shape()[-1].value

        with tf.name_scope(name) as scope:
            kernel = tf.get_variable(scope + "w",
                                     shape=[kh, kw, n_in, n_out],
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer_conv2d())
            conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
            bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
            biases = tf.Variable(bias_init_val, trainable=True, name='b')
            z = tf.nn.bias_add(conv, biases)
            activation = tf.nn.relu(z, name=scope)
            p += [kernel, biases]
            return activation

    def fc_op(self, input_op, name, n_out, p):
        n_in = input_op.get_shape()[-1].value

        with tf.name_scope(name) as scope:
            kernel = tf.get_variable(scope + "w",
                                     shape=[n_in, n_out],
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='b')
            activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
            p += [kernel, biases]
            return activation

    def mpool_op(self, input_op, name, kh, kw, dh, dw):
        return tf.nn.max_pool(input_op,
                              ksize=[1, kh, kw, 1],
                              strides=[1, dh, dw, 1],
                              padding='SAME',
                              name=name)

    def inference_op(self, input_op, keep_prob):
        p = []

        # block 1 -- outputs 112x112x64
        conv1_1 = self.conv_op(input_op, name="conv1_1", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
        conv1_2 = self.conv_op(conv1_1, name="conv1_2", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
        pool1 = self.mpool_op(conv1_2, name="pool1", kh=2, kw=2, dw=2, dh=2)

        # block 2 -- outputs 56x56x128
        conv2_1 = self.conv_op(pool1, name="conv2_1", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
        conv2_2 = self.conv_op(conv2_1, name="conv2_2", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
        pool2 = self.mpool_op(conv2_2, name="pool2", kh=2, kw=2, dh=2, dw=2)

        # # block 3 -- outputs 28x28x256
        conv3_1 = self.conv_op(pool2, name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
        conv3_2 = self.conv_op(conv3_1, name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
        conv3_3 = self.conv_op(conv3_2, name="conv3_3", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
        pool3 = self.mpool_op(conv3_3, name="pool3", kh=2, kw=2, dh=2, dw=2)

        # block 4 -- outputs 14x14x512
        conv4_1 = self.conv_op(pool3, name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
        conv4_2 = self.conv_op(conv4_1, name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
        conv4_3 = self.conv_op(conv4_2, name="conv4_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
        pool4 = self.mpool_op(conv4_3, name="pool4", kh=2, kw=2, dh=2, dw=2)

        # block 5 -- outputs 7x7x512
        conv5_1 = self.conv_op(pool4, name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
        conv5_2 = self.conv_op(conv5_1, name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
        conv5_3 = self.conv_op(conv5_2, name="conv5_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
        pool5 = self.mpool_op(conv5_3, name="pool5", kh=2, kw=2, dw=2, dh=2)

        # flatten
        shp = pool5.get_shape()
        flattened_shape = shp[1].value * shp[2].value * shp[3].value
        resh1 = tf.reshape(pool5, [-1, flattened_shape], name="resh1")

        # fully connected
        fc6 = self.fc_op(resh1, name="fc6", n_out=4096, p=p)
        fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc6_drop")

        fc7 = self.fc_op(fc6_drop, name="fc7", n_out=4096, p=p)
        fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")

        fc_out = self.fc_op(fc7_drop, name="fc8", n_out=1000, p=p)
        predict_prob = tf.nn.softmax(fc_out)
        predict_class = tf.argmax(predict_prob, 1)
        return predict_class, predict_prob, fc_out, p
