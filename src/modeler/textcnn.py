# -*- coding: utf-8 -*-
import tensorflow as tf

from modeler.tfmodel import TFModel


class TextCNNModel(TFModel):
    def __init__(self, filter_sizes, num_filters, num_classes, learning_rate, batch_size, decay_steps, decay_rate,
                 sequence_length, vocab_size, embed_size,
                 is_training, initializer=tf.random_normal_initializer(stddev=0.1), multi_label_flag=False,
                 clip_gradients=5.0, decay_rate_big=0.50):
        """init all hyperparameter here"""
        # set hyperparamter
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.is_training = is_training
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")  # ADD learning_rate
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * decay_rate_big)
        self.filter_sizes = filter_sizes  # it is a list of int. e.g. [3,4,5]
        self.num_filters = num_filters
        self.initializer = initializer
        self.num_filters_total = self.num_filters * len(filter_sizes)  # how many filters totally.
        self.multi_label_flag = multi_label_flag
        self.clip_gradients = clip_gradients
        self.decay_steps, self.decay_rate = decay_steps, decay_rate
        super(TextCNNModel, self).__init__()

    def add_placeholder(self):
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")  # X
        self.input_y_multilabel = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y_multilabel")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.iter = tf.placeholder(tf.int32)  # training iteration
        self.tst = tf.placeholder(tf.bool)

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.b1 = tf.Variable(tf.ones([self.num_filters]) / 10)
        self.b2 = tf.Variable(tf.ones([self.num_filters]) / 10)
        self.instantiate_weights()

    def build(self):
        self.logits = self.inference()  # [None, self.label_size]. main computation graph is here.

    def cal_predict(self):
        self.possibility = tf.nn.sigmoid(self.logits)

    def cal_loss(self):
        if self.multi_label_flag:
            print("going to use multi label loss.")
            self.loss_val = self.loss_multilabel()
        else:
            print("going to use single label loss.")
            self.loss_val = self.loss()

    def recon_optimize(self):
        self.train_op = self.train()

    def cal_accuracy(self):
        if not self.multi_label_flag:
            self.predictions = tf.argmax(self.logits, 1, name="predictions")  # shape:[None,]
            print("self.predictions:", self.predictions)
            correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.input_y_multilabel)
            # tf.argmax(self.logits, 1)-->[batch_size]
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")  # shape=()

    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("embedding"):  # embedding matrix
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],
                                             initializer=self.initializer)  # [vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            self.W_projection = tf.get_variable("W_projection", shape=[self.num_filters_total, self.num_classes],
                                                initializer=self.initializer)  # [embed_size,label_size]
            self.b_projection = tf.get_variable("b_projection",
                                                shape=[self.num_classes])  # [label_size] #ADD 2017.06.09

    def inference(self):
        """main computation graph here: 1.embedding-->2.CONV-BN-RELU-MAX_POOLING-->3.linear classifier"""
        # 1.=====>get emebedding of words in the sentence
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding, self.input_x)
        # [None,sentence_length,embed_size]
        self.sentence_embeddings_expanded = tf.expand_dims(self.embedded_words, -1)
        # [None,sentence_length,embed_size,1). expand dimension so meet input requirement of 2d-conv

        # 2.=====>loop each filter size. for each filter,
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("convolution-pooling-%s" % filter_size):
                # ====>a.create filter
                filter = tf.get_variable("filter-%s" % filter_size,
                                         [filter_size, self.embed_size, 1, self.num_filters],
                                         initializer=self.initializer)
                # input data format:NHWC:[batch, height, width, channels];output:4-D
                conv = tf.nn.conv2d(self.sentence_embeddings_expanded, filter, strides=[1, 1, 1, 1],
                                    padding="VALID",
                                    name="conv")
                # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
                conv, self.update_ema = self.batchnorm(conv, self.tst, self.iter, self.b1)
                # ====>c. apply nolinearity
                b = tf.get_variable("b-%s" % filter_size, [self.num_filters])  # ADD 2017-06-09
                h = tf.nn.relu(tf.nn.bias_add(conv, b), "relu")
                # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`
                # ====>. max-pooling.  value: A 4-D `Tensor` with shape `[batch, height, width, channels]
                # ksize: A list of ints that has length >= 4.  The size of the window for each dimension of the input tensor.
                # strides: A list of ints that has length >= 4.
                # The stride of the sliding window for each dimension of the input tensor.
                pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID',
                                        name="pool")
                # shape:[batch_size, 1, 1, num_filters].max_pool:performs the max pooling on the input.
                pooled_outputs.append(pooled)
        # 3.=====>combine all pooled features, and flatten the feature.output' shape is a [1,None]
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total])
        # shape should be:[None,num_filters_total]. here this operation has some result as tf.sequeeze().
        # e.g. x's shape:[3,3];tf.reshape(-1,x) & (3, 3)---->(1,9)

        # 4.=====>add dropout: use tf.nn.dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat,
                                        keep_prob=self.dropout_keep_prob)  # [None,num_filters_total]
        self.h_drop = tf.layers.dense(self.h_drop, self.num_filters_total, activation=tf.nn.tanh, use_bias=True)
        # 5. logits(use linear layer)and predictions(argmax)
        with tf.name_scope("output"):
            logits = tf.matmul(self.h_drop, self.W_projection) + self.b_projection
            # shape:[None, self.num_classes]==tf.matmul([None,self.embed_size],[self.embed_size,self.num_classes])
        return logits

    def batchnorm(self, Ylogits, is_test, iteration, offset, convolutional=False):
        # mnist_4.1_batchnorm_five_layers_relu.py#L89
        """
        batch normalization: keep moving average of mean and variance. use it as value
        for BN when training. when prediction, use value from that batch.
        :param Ylogits:
        :param is_test:
        :param iteration:
        :param offset:
        :param convolutional:
        :return:
        """
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration)
        # adding the iteration prevents from averaging across non-existing iterations
        bnepsilon = 1e-5
        if convolutional:
            mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
        else:
            mean, variance = tf.nn.moments(Ylogits, [0])
        update_moving_averages = exp_moving_avg.apply([mean, variance])
        m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
        v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
        Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
        return Ybn, update_moving_averages

    def loss_multilabel(self, l2_lambda=0.0001):
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel, logits=self.logits)
            print("sigmoid_cross_entropy_with_logits.losses:", losses)  # shape=(?, 1999).
            losses = tf.reduce_sum(losses, axis=1)  # shape=(?,). loss for all data in the batch
            loss = tf.reduce_mean(losses)  # shape=().   average loss in the batch
            l2_losses = tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        return loss

    def loss(self, l2_lambda=0.0001):  # 0.001
        with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            # sigmoid_cross_entropy_with_logits.
            loss = tf.reduce_mean(losses)  # print("2.loss.loss:", loss) #shape=()
            l2_losses = tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                   self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
                                                   learning_rate=learning_rate, optimizer="Adam",
                                                   clip_gradients=self.clip_gradients)
        return train_op
