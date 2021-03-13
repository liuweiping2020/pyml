import math

import tensorflow as tf

from modeler.tfmodel import TFModel


class VDCNN(TFModel):
    def __init__(self, num_classes, sequence_max_length=1024, num_quantized_chars=69, embedding_size=16,
                 depth=9, downsampling_type='maxpool', use_he_uniform=True, optional_shortcut=False):
        self.num_classes = num_classes
        self.sequence_max_length = sequence_max_length
        self.num_quantized_chars = num_quantized_chars
        self.embedding_size = embedding_size,
        # self.depth = 9
        self.downsampling_type = downsampling_type
        self.use_he_uniform = use_he_uniform
        self.optional_shortcut = optional_shortcut
        self.he_normal = tf.keras.initializers.he_normal()
        self.regularizer = tf.contrib.layers.l2_regularizer(1e-4)

        # Depth to No. Layers
        if depth == 9:
            self.num_layers = [2, 2, 2, 2]
        elif depth == 17:
            self.num_layers = [4, 4, 4, 4]
        elif depth == 29:
            self.num_layers = [10, 10, 4, 4]
        elif depth == 49:
            self.num_layers = [16, 16, 10, 6]
        else:
            raise ValueError('depth=%g is a not a valid setting!' % depth)

    def add_placeholder(self):
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_max_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")
        self.is_training = tf.placeholder(tf.bool)
        pass

    def build(self):
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            if self.use_he_uniform:
                self.embedding_W = tf.get_variable(name='lookup_W',
                                                   shape=[self.num_quantized_chars, self.embedding_size],
                                                   initializer=tf.keras.initializers.he_uniform())
            else:
                self.embedding_W = tf.Variable(
                    tf.random_uniform([self.num_quantized_chars, self.embedding_size], -1.0, 1.0),
                    name="embedding_W")
            self.embedded_characters = tf.nn.embedding_lookup(self.embedding_W, self.input_x)
            print("-" * 20)
            print("Embedded Lookup:", self.embedded_characters.get_shape())
            print("-" * 20)

        self.layers = []

        # Temp(First) Conv Layer
        with tf.variable_scope("temp_conv") as scope:
            filter_shape = [3, self.embedding_size, 64]
            W = tf.get_variable(name='W_1', shape=filter_shape,
                                initializer=self.he_normal,
                                regularizer=self.regularizer)
            inputs = tf.nn.conv1d(self.embedded_characters, W, stride=1, padding="SAME")
            # inputs = tf.nn.relu(inputs)
        print("Temp Conv", inputs.get_shape())
        self.layers.append(inputs)

        for i in range(self.num_layers[0]):
            if i < self.num_layers[0] - 1 and self.optional_shortcut:
                shortcut = self.layers[-1]
            else:
                shortcut = None
            conv_block = self.Convolutional_Block(inputs=self.layers[-1], shortcut=shortcut, num_filters=64,
                                                  is_training=self.is_training, name=str(i + 1))
            self.layers.append(conv_block)
        pool1 = self.downsampling(self.layers[-1], downsampling_type=self.downsampling_type, name='pool1',
                                  optional_shortcut=self.optional_shortcut, shortcut=self.layers[-2])
        self.layers.append(pool1)
        print("Pooling:", pool1.get_shape())

        for i in range(self.num_layers[1]):
            if i < self.num_layers[1] - 1 and self.optional_shortcut:
                shortcut = self.layers[-1]
            else:
                shortcut = None
            conv_block = self.Convolutional_Block(inputs=self.layers[-1], shortcut=shortcut, num_filters=128,
                                                  is_training=self.is_training, name=str(i + 1))
            self.layers.append(conv_block)
        pool2 = self.downsampling(self.layers[-1], downsampling_type=self.downsampling_type, name='pool2',
                                  optional_shortcut=self.optional_shortcut, shortcut=self.layers[-2])
        self.layers.append(pool2)
        print("Pooling:", pool2.get_shape())

        for i in range(self.num_layers[2]):
            if i < self.num_layers[2] - 1 and self.optional_shortcut:
                shortcut = self.layers[-1]
            else:
                shortcut = None
            conv_block = self.Convolutional_Block(inputs=self.layers[-1], shortcut=shortcut, num_filters=256,
                                                  is_training=self.is_training, name=str(i + 1))
            self.layers.append(conv_block)
        pool3 = self.downsampling(self.layers[-1], downsampling_type=self.downsampling_type, name='pool3',
                                  optional_shortcut=self.optional_shortcut, shortcut=self.layers[-2])
        self.layers.append(pool3)
        print("Pooling:", pool3.get_shape())

        for i in range(self.num_layers[3]):
            if i < self.num_layers[3] - 1 and self.optional_shortcut:
                shortcut = self.layers[-1]
            else:
                shortcut = None
            conv_block = self.Convolutional_Block(inputs=self.layers[-1], shortcut=shortcut, num_filters=512,
                                                  is_training=self.is_training, name=str(i + 1))
            self.layers.append(conv_block)

        self.k_pooled = tf.nn.top_k(tf.transpose(self.layers[-1], [0, 2, 1]), k=8, name='k_pool', sorted=False)[0]
        print("8-maxpooling:", self.k_pooled.get_shape())
        self.flatten = tf.reshape(self.k_pooled, (-1, 512 * 8))

        # fc1
        with tf.variable_scope('fc1'):
            w = tf.get_variable('w', [self.flatten.get_shape()[1], 2048], initializer=self.he_normal,
                                regularizer=self.regularizer)
            b = tf.get_variable('b', [2048], initializer=tf.constant_initializer(1.0))
            out = tf.matmul(self.flatten, w) + b
            self.fc1 = tf.nn.relu(out)

        # fc2
        with tf.variable_scope('fc2'):
            w = tf.get_variable('w', [self.fc1.get_shape()[1], 2048], initializer=self.he_normal,
                                regularizer=self.regularizer)
            b = tf.get_variable('b', [2048], initializer=tf.constant_initializer(1.0))
            out = tf.matmul(self.fc1, w) + b
            self.fc2 = tf.nn.relu(out)

        # fc3
        with tf.variable_scope('fc3'):
            w = tf.get_variable('w', [self.fc2.get_shape()[1], self.num_classes], initializer=self.he_normal,
                                regularizer=self.regularizer)
            b = tf.get_variable('b', [self.num_classes], initializer=tf.constant_initializer(1.0))
            self.fc3 = tf.matmul(self.fc2, w) + b

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            self.predictions = tf.argmax(self.fc3, 1, name="predictions")
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.fc3, labels=self.input_y)
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.loss = tf.reduce_mean(losses) + sum(regularization_losses)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def Convolutional_Block(self, inputs, shortcut, num_filters, name, is_training):
        print("-" * 20)
        print("Convolutional Block", str(num_filters), name)
        print("-" * 20)
        with tf.variable_scope("conv_block_" + str(num_filters) + "_" + name):
            for i in range(2):
                with tf.variable_scope("conv1d_%s" % str(i)):
                    filter_shape = [3, inputs.get_shape()[2], num_filters]
                    W = tf.get_variable(name='W', shape=filter_shape,
                                        initializer=self.he_normal,
                                        regularizer=self.regularizer)
                    inputs = tf.nn.conv1d(inputs, W, stride=1, padding="SAME")
                    inputs = tf.layers.batch_normalization(inputs=inputs, momentum=0.997, epsilon=1e-5,
                                                           center=True, scale=True, training=is_training)
                    inputs = tf.nn.relu(inputs)
                    print("Conv1D:", inputs.get_shape())
        print("-" * 20)
        if shortcut is not None:
            print("-" * 5)
            print("Optional Shortcut:", shortcut.get_shape())
            print("-" * 5)
            return inputs + shortcut
        return inputs

    # Three types of downsampling methods described by paper
    def downsampling(self, inputs, downsampling_type, name, optional_shortcut=False, shortcut=None):
        # k-maxpooling
        if downsampling_type == 'k-maxpool':
            k = math.ceil(int(inputs.get_shape()[1]) / 2)
            pool = tf.nn.top_k(tf.transpose(inputs, [0, 2, 1]), k=k, name=name, sorted=False)[0]
            pool = tf.transpose(pool, [0, 2, 1])
        # Linear
        elif downsampling_type == 'linear':
            pool = tf.layers.conv1d(inputs=inputs, filters=inputs.get_shape()[2], kernel_size=3,
                                    strides=2, padding='same', use_bias=False)
        # Maxpooling
        else:
            pool = tf.layers.max_pooling1d(inputs=inputs, pool_size=3, strides=2, padding='same', name=name)
        if optional_shortcut:
            shortcut = tf.layers.conv1d(inputs=shortcut, filters=shortcut.get_shape()[2], kernel_size=1,
                                        strides=2, padding='same', use_bias=False)
            print("-" * 5)
            print("Optional Shortcut:", shortcut.get_shape())
            print("-" * 5)
            pool += shortcut
        pool = self.fixed_padding(inputs=pool)
        return tf.layers.conv1d(inputs=pool, filters=pool.get_shape()[2] * 2, kernel_size=1,
                                strides=1, padding='valid', use_bias=False)

    def fixed_padding(self, inputs, kernel_size=3):
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [0, 0]])
        return padded_inputs
