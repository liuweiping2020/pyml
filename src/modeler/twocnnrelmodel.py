# -*- coding: utf-8 -*-
import tensorflow as tf

from modeler.tfmodel import TFModel


class TwoCNNRelModel(TFModel):
    def __init__(self, filter_sizes, num_filters, num_classes, learning_rate, batch_size, decay_steps, decay_rate,
                 sequence_length, vocab_size,
                 embed_size, is_training, initializer=tf.random_normal_initializer(stddev=0.1)):
        """init all hyperparameter here"""
        # set hyperparamter
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.is_training = is_training
        self.learning_rate = learning_rate
        self.filter_sizes = filter_sizes  # it is a list of int. e.g. [3,4,5]
        self.num_filters = num_filters
        self.initializer = initializer
        self.num_filters_total = self.num_filters * len(filter_sizes)  # how many filters totally.
        self.decay_steps, self.decay_rate = decay_steps, decay_rate
        super(TwoCNNRelModel,self).__init__()

    def add_placeholder(self):
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")  # X: first sentence
        self.input_x2 = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x2")
        # X: second sentence
        self.input_y = tf.placeholder(tf.int32, [None, ], name="input_y")
        # y: 0 or 1. 1 means two sentences related to each other;0 means no relation.
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.instantiate_weights()

    def build(self):
        self.logits = self.inference()  # [None, self.label_size]. main computation graph is here.

    def cal_loss(self):
        self.loss_val = self.loss()

    def recon_optimize(self):
        self.train_op = self.train()

    def cal_predict(self):
        self.predictions = tf.argmax(self.logits, 1, name="predictions")  # shape:[None,]

    def cal_accuracy(self):
        correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.input_y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")  # shape=()

    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("embedding"):  # embedding matrix
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],
                                             initializer=self.initializer)
            self.W_projection = tf.get_variable("W_projection",
                                                shape=[self.num_filters_total * 2, self.num_classes],
                                                initializer=self.initializer)
            # [embed_size,label_size]
            self.b_projection = tf.get_variable("b_projection", shape=[self.num_classes])  # [label_size]

    def inference(self):
        """main computation graph here:
        1. embeddding layers, 2.convolutional layer, 3.max-pooling, 4.softmax layer."""
        # 1.=====>get emebedding of words in the sentence
        self.embedded_words1 = tf.nn.embedding_lookup(self.Embedding, self.input_x)
        # [None,sentence_length,embed_size]
        self.sentence_embeddings_expanded1 = tf.expand_dims(self.embedded_words1, -1)
        # [None,sentence_length,embed_size,1). expand dimension so meet input requirement of 2d-conv
        self.embedded_words2 = tf.nn.embedding_lookup(self.Embedding, self.input_x2)
        # [None,sentence_length,embed_size]
        self.sentence_embeddings_expanded2 = tf.expand_dims(self.embedded_words2, -1)
        # [None,sentence_length,embed_size,1). expand dimension so meet input requirement of 2d-conv
        # 2.1 get features of sentence1
        h1 = self.conv_relu_pool_dropout(self.sentence_embeddings_expanded1, name_scope_prefix="s1")
        # [None,num_filters_total]
        # 2.2 get features of sentence2
        h2 = self.conv_relu_pool_dropout(self.sentence_embeddings_expanded2, name_scope_prefix="s2")
        # [None,num_filters_total]
        # 3. concat features
        h = tf.concat([h1, h2], axis=1)  # [None,num_filters_total*2]
        # 4. logits(use linear layer)and predictions(argmax)
        with tf.name_scope("output"):
            logits = tf.matmul(h, self.W_projection) + self.b_projection
        return logits

    def conv_relu_pool_dropout(self, sentence_embeddings_expanded, name_scope_prefix=None):
        # 1.loop each filter size.
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope(name_scope_prefix + "convolution-pooling-%s" % filter_size):
                # ====>a.create filter
                filter = tf.get_variable(name_scope_prefix + "filter-%s" % filter_size,
                                         [filter_size, self.embed_size, 1, self.num_filters],
                                         initializer=self.initializer)
                conv = tf.nn.conv2d(sentence_embeddings_expanded, filter, strides=[1, 1, 1, 1], padding="VALID",
                                    name="conv")
                # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
                # ====>c. apply nolinearity
                b = tf.get_variable(name_scope_prefix + "b-%s" % filter_size, [self.num_filters])
                h = tf.nn.relu(tf.nn.bias_add(conv, b), "relu")
                pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID', name="pool")
                # shape:[batch_size, 1, 1, num_filters].max_pool:performs the max pooling on the input.
                pooled_outputs.append(pooled)
        # 2.=====>combine all pooled features, and flatten the feature.output' shape is a [1,None]
        h_pool = tf.concat(pooled_outputs, 3)
        # shape:[batch_size, 1, 1, num_filters_total].
        h_pool_flat = tf.reshape(h_pool, [-1, self.num_filters_total])
        # 3.=====>add dropout: use tf.nn.dropout
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, keep_prob=self.dropout_keep_prob)
        return h_drop

    def loss(self, l2_lambda=0.0001):  # 0.001
        with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            loss = tf.reduce_mean(losses)  # print("2.loss.loss:", loss) #shape=()
            l2_losses = tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        return loss

    def loss_multilabel(self, l2_lambda=0.001):  # this loss function is for multi-label classification
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel,
                                                             logits=self.logits)
            print("sigmoid_cross_entropy_with_logits.losses:", losses)  # shape=(?, 1999)
            losses = tf.reduce_sum(losses, axis=1)  # shape=(?,)
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                   self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
                                                   learning_rate=learning_rate, optimizer="Adam")
        return train_op

