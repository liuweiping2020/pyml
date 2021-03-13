# -*- coding: utf-8 -*-
import copy

import tensorflow as tf

from modeler.tfmodel import TFModel


class CNNRCNNModel(TFModel):
    def __init__(self, filter_sizes, num_filters, num_classes, learning_rate, batch_size, decay_steps, decay_rate,
                 sequence_length, vocab_size, embed_size,
                 is_training, initializer=tf.random_normal_initializer(stddev=0.1), multi_label_flag=False):
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
        self.num_filters_total = self.num_filters * len(filter_sizes)
        self.multi_label_flag = multi_label_flag
        self.hidden_size = embed_size
        self.activation = tf.nn.tanh
        self.decay_steps, self.decay_rate = decay_steps, decay_rate
        super(CNNRCNNModel,self).__init__()

    def add_placeholder(self):
        # add placeholder (X,label)
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")  # X
        self.input_y = tf.placeholder(tf.int32, [None, ], name="input_y")  # y:[None,num_classes]
        self.input_y_multilabel = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y_multilabel")
        # y:[None,num_classes]. this is for multi-label classification only.
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.instantiate_weights_cnn()
        self.instantiate_weights_rcnn()

    def build(self):
        self.logits = self.inference()  # [None, self.label_size]. main computation graph is here.

    def cal_loss(self):
        if self.multi_label_flag:
            print("going to use multi label loss.")
            self.loss_val = self.loss_multilabel()
        else:
            print("going to use single label loss.")
            self.loss_val = self.loss()

    def recon_optimize(self):
        self.train_op = self.train()

    def cal_predict(self):
        self.predictions = tf.argmax(self.logits, 1, name="predictions")  # shape:[None,]

    def cal_accuracy(self):
        if not self.multi_label_flag:
            correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.input_y)
            # tf.argmax(self.logits, 1)-->[batch_size]
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")  # shape=()
        else:
            self.accuracy = tf.constant(0.5)

    def instantiate_weights_cnn(self):
        """define all weights here"""
        with tf.name_scope("projection_cnn"):  # embedding matrix
            self.W_projection_cnn = tf.get_variable("W_projection_cnn",
                                                    shape=[self.num_filters_total, self.num_classes],
                                                    initializer=self.initializer)  # [embed_size,label_size]
            self.b_projection_cnn = tf.get_variable("b_projection_cnn", shape=[self.num_classes])  # [label_size]

    def instantiate_weights_rcnn(self):
        """define all weights here"""
        self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],
                                         initializer=self.initializer)  # [vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)

        with tf.name_scope("weights_rcnn"):  # embedding matrix
            self.left_side_first_word = tf.get_variable("left_side_first_word",
                                                        shape=[self.batch_size, self.embed_size],
                                                        initializer=self.initializer)
            self.right_side_last_word = tf.get_variable("right_side_last_word",
                                                        shape=[self.batch_size, self.embed_size],
                                                        initializer=self.initializer)
            self.W_l = tf.get_variable("W_l", shape=[self.embed_size, self.embed_size],
                                       initializer=self.initializer)
            self.W_r = tf.get_variable("W_r", shape=[self.embed_size, self.embed_size],
                                       initializer=self.initializer)
            self.W_sl = tf.get_variable("W_sl", shape=[self.embed_size, self.embed_size],
                                        initializer=self.initializer)
            self.W_sr = tf.get_variable("W_sr", shape=[self.embed_size, self.embed_size],
                                        initializer=self.initializer)

            self.W_projection_rcnn = tf.get_variable("W_projection", shape=[self.hidden_size * 3, self.num_classes],
                                                     initializer=self.initializer)  # [embed_size,label_size]
            self.b_projection_rcnn = tf.get_variable("b_projection", shape=[self.num_classes])  # [label_size]

    def inference1(self):
        """main computation graph here: 1.embedding-->2.average-->3.linear classifier"""
        # 1.=====>get emebedding of words in the sentence
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding, self.input_x)
        # [None,sentence_length,embed_size]
        self.sentence_embeddings_expanded = tf.expand_dims(self.embedded_words, -1)
        # [None,sentence_length,embed_size,1). expand dimension so meet input requirement of 2d-conv

        # 2.=====>loop each filter size. for each filter, do:convolution-pooling layer(a.create filters,
        # b.conv,c.apply nolinearity,d.max-pooling)--->
        # you can use:tf.nn.conv2d;tf.nn.relu;tf.nn.max_pool; feature shape is 4-d. feature is a new variable
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("convolution-pooling-%s" % filter_size):
                filter = tf.get_variable("filter-%s" % filter_size,
                                         [filter_size, self.embed_size, 1, self.num_filters],
                                         initializer=self.initializer)
                conv = tf.nn.conv2d(self.sentence_embeddings_expanded, filter, strides=[1, 1, 1, 1],
                                    padding="VALID",
                                    name="conv")
                b = tf.get_variable("b-%s" % filter_size, [self.num_filters])
                h = tf.nn.relu(tf.nn.bias_add(conv, b), "relu")
                pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID',
                                        name="pool")
                pooled_outputs.append(pooled)
        # 3.=====>combine all pooled features, and flatten the feature.output' shape is a [1,None]
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total])
        # shape should be:[None,num_filters_total]. here this operation has some result as tf.sequeeze().
        # e.g. x's shape:[3,3];tf.reshape(-1,x) & (3, 3)---->(1,9)

        # 4.=====>add dropout: use tf.nn.dropout
        with tf.name_scope("dropout_cnn"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, keep_prob=self.dropout_keep_prob)
            # [None,num_filters_total]

        # 5. logits(use linear layer)and predictions(argmax)
        with tf.name_scope("output_cnn"):
            logits = tf.matmul(self.h_drop, self.W_projection_cnn) + self.b_projection_cnn
            # shape:[None, self.num_classes]==tf.matmul([None,self.embed_size],[self.embed_size,self.num_classes])
        return logits

    def get_context_left(self, context_left, embedding_previous):
        """
        :param context_left:[batch_size,self.embed_size]
        :param embedding_previous:[batch_size,self.embed_size]
        :return: output:[None,embed_size]
        """
        left_c = tf.matmul(context_left, self.W_l)
        # context_left:[batch_size,embed_size];W_l:[embed_size,embed_size]
        left_e = tf.matmul(embedding_previous, self.W_sl)  # embedding_previous;[batch_size,embed_size]
        left_h = left_c + left_e
        context_left = self.activation(left_h)
        return context_left

    def get_context_right(self, context_right, embedding_afterward):
        """
        :param context_right:[batch_size,self.embed_size]
        :param embedding_afterward:[batch_size,self.embed_size]
        :return: output:[batch_size,embed_size]
        """
        right_c = tf.matmul(context_right, self.W_r)
        right_e = tf.matmul(embedding_afterward, self.W_sr)
        right_h = right_c + right_e
        context_right = self.activation(right_h)
        return context_right

    def conv_layer_with_recurrent_structure(self):
        """
        input:self.embedded_words:[None,sentence_length,embed_size]
        :return: shape:[None,sentence_length,embed_size*3]
        """
        # 1. get splitted list of word embeddings
        embedded_words_split = tf.split(self.embedded_words, self.sequence_length, axis=1)
        # sentence_length个[None,1,embed_size]
        embedded_words_squeezed = [tf.squeeze(x, axis=1) for x in embedded_words_split]
        # sentence_length个[None,embed_size]
        embedding_previous = self.left_side_first_word
        context_left_previous = tf.zeros((self.batch_size, self.embed_size))
        # 2. get list of context left
        context_left_list = []
        for i, current_embedding_word in enumerate(embedded_words_squeezed):
            # sentence_length个[None,embed_size]
            context_left = self.get_context_left(context_left_previous, embedding_previous)
            # [None,embed_size]
            context_left_list.append(context_left)  # append result to list
            embedding_previous = current_embedding_word  # assign embedding_previous
            context_left_previous = context_left  # assign context_left_previous
        # 3. get context right
        embedded_words_squeezed2 = copy.copy(embedded_words_squeezed)
        embedded_words_squeezed2.reverse()
        embedding_afterward = self.right_side_last_word
        context_right_afterward = tf.zeros((self.batch_size, self.embed_size))
        context_right_list = []
        for j, current_embedding_word in enumerate(embedded_words_squeezed2):
            context_right = self.get_context_right(context_right_afterward, embedding_afterward)
            context_right_list.append(context_right)
            embedding_afterward = current_embedding_word
            context_right_afterward = context_right
        # 4.ensemble left,embedding,right to output
        output_list = []
        for index, current_embedding_word in enumerate(embedded_words_squeezed):
            representation = tf.concat(
                [context_left_list[index], current_embedding_word, context_right_list[index]], axis=1)
            # print(i,"representation:",representation)
            output_list.append(representation)  # shape:sentence_length个[None,embed_size*3]
        # 5. stack list to a tensor
        # print("output_list:",output_list) #(3, 5, 8, 100)
        output = tf.stack(output_list, axis=1)  # shape:[None,sentence_length,embed_size*3]
        # print("output:",output)
        return output

    def inference2(self):
        """main computation graph here: 1. embeddding layer, 2.Bi-LSTM layer, 3.max pooling, 4.FC layer 5.softmax """
        # 1.get emebedding of words in the sentence
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding, self.input_x)
        # shape:[None,sentence_length,embed_size]
        # 2. Bi-lstm layer
        output_conv = self.conv_layer_with_recurrent_structure()  # shape:[None,sentence_length,embed_size*3]
        # 3. max pooling
        # print("output_conv:",output_conv) #(3, 5, 8, 100)
        output_pooling = tf.reduce_max(output_conv, axis=1)  # shape:[None,embed_size*3]
        # print("output_pooling:",output_pooling) #(3, 8, 100)
        # 4. logits(use linear layer)
        with tf.name_scope("dropout_rcnn"):
            h_drop = tf.nn.dropout(output_pooling, keep_prob=self.dropout_keep_prob)  # [None,embed_size*3]

            # with tf.name_scope("output"): #inputs: A `Tensor` of shape `[batch_size, dim]`.
            #  The forward activations of the input network.
            logits = tf.matmul(h_drop, self.W_projection_rcnn) + self.b_projection_rcnn
            # [batch_size,num_classes]
        return logits

    def inference(self):
        weight1 = tf.get_variable("weight1", shape=(), initializer=self.initializer)
        self.p_weight1 = tf.nn.sigmoid(weight1)
        self.p_weight2 = 1.0 - self.p_weight1
        logits1 = self.inference1()
        logits2 = self.inference2()
        logits = self.p_weight1 * logits1 + self.p_weight2 * logits2
        return logits

    def loss(self, l2_lambda=0.0001):  # 0.001
        with tf.name_scope("loss"):
            # input: `logits`:[batch_size, num_classes], and `labels`:[batch_size]
            # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            loss = tf.reduce_mean(losses)  # print("2.loss.loss:", loss) #shape=()
            l2_losses = tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        return loss

    def loss_multilabel(self, l2_lambda=0.00001):
        with tf.name_scope("loss"):
            # input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
            # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            # input_y:shape=(?, 1999); logits:shape=(?, 1999)
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel, logits=self.logits)
            print("sigmoid_cross_entropy_with_logits.losses:", losses)  # shape=(?, 1999).
            losses = tf.reduce_sum(losses, axis=1)
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


